# -*- coding: UTF-8 -*-
"""
Minimal TIGER model adapter for ReChorus.

This file provides a wrapper around a conditional generation model (e.g. T5)
and adapts it to ReChorus' Model/Dataset/parse_model_args conventions.

Notes:
- We import transformers lazily inside the constructor to avoid hard import
  time dependency at module import.
- The implementation assumes `corpus.item_codes[item_id]` yields a token id or
  a short list of token ids representing that item. The Dataset below tries to
  concatenate/flatten item codes into an input sequence and pads/truncates to
  `args.max_len`.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


class TIGER(SequentialModel):
    """Adapter for TIGER-style generation model embedded in ReChorus.

    Key changes vs. original standalone TIGER:
    - Use existing `SeqReader` (so we get user history, positions, etc.).
    - Perform all item->token_code preprocessing lazily inside Dataset._get_feed_dict
      (load mapping from a provided .npy at runtime via --tiger_code_path) so there
      is no external preprocessing step.
    - Default runner is `BaseRunner` to keep ReChorus training/eval flow.
    """

    # Use TIGERReader by default so we can normalize .npy codes for datasets like Grocery
    reader, runner = 'TIGERReader', 'TIGERRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--vocab_size', type=int, default=1025, help='Vocab size for generation model')
        parser.add_argument('--pad_token_id', type=int, default=0, help='Pad token id')
        parser.add_argument('--eos_token_id', type=int, default=0, help='EOS token id')
        parser.add_argument('--max_len', type=int, default=20, help='Max input length (in tokens)')
        parser.add_argument('--beam_size', type=int, default=30, help='Beam size for generation')
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--num_decoder_layers', type=int, default=4)
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--d_ff', type=int, default=1024)
        parser.add_argument('--num_heads', type=int, default=6)
        parser.add_argument('--d_kv', type=int, default=64)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--feed_forward_proj', type=str, default='relu')

        # ===== Small improvement for the course project: Label Smoothing =====
        # eps=0.0 keeps the original training objective.
        parser.add_argument('--label_smoothing', type=float, default=0.0,
                            help='Label smoothing epsilon for seq2seq token loss (0.0 disables).')

        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.args = args
        self.pad_token_id = args.pad_token_id
        self.eos_token_id = args.eos_token_id

        self.max_len = args.max_len
        self.beam_size = args.beam_size

        # Keep corpus reference but don't rely on corpus.item_codes existing.
        self.corpus = corpus
        self.item_codes = None  # will be loaded lazily from args.tiger_code_path or built from item ids

        # If a code mapping path is provided, remember it for lazy load
        self.tiger_code_path = getattr(args, 'tiger_code_path', '')

        # Additional TIGER-specific options (defaults chosen to match original project)
        # codebook_k: number of entries per codebook (default 256 in original TIGER)
        # num_codebooks: number of codebooks per item (default 4 in generate_code.py)
        self.codebook_k = getattr(args, 'codebook_k', 256)
        self.num_codebooks = getattr(args, 'num_codebooks', 4)

        # Try to prepare item_codes now if possible: prefer corpus.item_codes, then tiger_code_path
        try:
            n_items = int(getattr(self.corpus, 'n_items', 0))
        except Exception:
            n_items = 0

        # If corpus already has item_codes (e.g., preprocessed), ensure it's in the expected format
        if hasattr(self.corpus, 'item_codes') and getattr(self.corpus, 'item_codes', None) is not None:
            logging.info('TIGER: using corpus.item_codes already present')
        else:
            # Try to eagerly load code mapping from provided path and normalize it to global token ids
            if self.tiger_code_path:
                try:
                    import numpy as _np
                    loaded = _np.load(self.tiger_code_path, allow_pickle=True)
                    logging.info('TIGER: loaded tiger_code_path %s with shape %s',
                                 self.tiger_code_path, getattr(loaded, 'shape', None))
                    normalized = []

                    # if loaded is a dict-like saved as npy (item->code), try to handle
                    if isinstance(loaded, dict) or (
                        hasattr(loaded, 'dtype') and loaded.dtype == object and loaded.ndim == 1
                        and isinstance(loaded.tolist()[0], (list, tuple))
                    ):
                        for entry in loaded.tolist():
                            if entry is None:
                                normalized.append([])
                            else:
                                normalized.append(list(entry))
                    else:
                        try:
                            arr = _np.array(loaded)
                            if arr.ndim == 2:
                                for row in arr.tolist():
                                    normalized.append(list(row))
                            elif arr.ndim == 1:
                                for v in arr.tolist():
                                    normalized.append([int(v)])
                            else:
                                for v in loaded:
                                    normalized.append(list(v) if hasattr(v, '__iter__') else [int(v)])
                        except Exception:
                            for v in loaded:
                                normalized.append(list(v) if hasattr(v, '__iter__') else [int(v)])

                    # Detect whether values are small per-codebook indices (< codebook_k) or global token ids.
                    max_val = max([max(row) if len(row) > 0 else -1 for row in normalized]) if len(normalized) > 0 else -1
                    if max_val >= 0 and max_val < self.codebook_k:
                        logging.info('TIGER: detected per-codebook indices; converting to global token ids using codebook_k=%d',
                                     self.codebook_k)
                        converted = []
                        for row in normalized:
                            codes = list(row)[:self.num_codebooks]
                            if len(codes) < self.num_codebooks:
                                codes = codes + [0] * (self.num_codebooks - len(codes))
                            offsets = [int(c) + i * self.codebook_k + 1 for i, c in enumerate(codes)]
                            converted.append(offsets)
                        self.corpus.item_codes = converted
                    else:
                        self.corpus.item_codes = [
                            list(row) if hasattr(row, '__iter__') and not isinstance(row, (str, bytes)) else [int(row)]
                            for row in normalized
                        ]

                    # If loaded length equals n_items+1 and dataset uses 1-based ids, shift: keep index i => item_id i
                    try:
                        if n_items and len(self.corpus.item_codes) == n_items + 1:
                            logging.info('TIGER: code array length == n_items+1, assuming 1-based item ids; dropping index 0 for 0-based corpus')
                            self.corpus.item_codes = self.corpus.item_codes[1:]
                    except Exception:
                        pass

                except Exception as e:
                    logging.warning('TIGER: failed to eagerly load tiger_code_path %s: %s', self.tiger_code_path, e)
                    self.corpus.item_codes = None
            else:
                # No code path given: build a deterministic mapping from item_id -> per-codebook indices then convert to offsets
                if n_items and n_items > 0:
                    logging.info('TIGER: building deterministic item_codes for %d items (num_codebooks=%d, codebook_k=%d)',
                                 n_items, self.num_codebooks, self.codebook_k)
                    built = []
                    for iid in range(n_items):
                        codes = [int((iid + i) % self.codebook_k) for i in range(self.num_codebooks)]
                        offsets = [c + i * self.codebook_k + 1 for i, c in enumerate(codes)]
                        built.append(offsets)
                    self.corpus.item_codes = built
                else:
                    self.corpus.item_codes = None

        # Ensure max_len is at least num_codebooks so codes are not truncated during generation
        try:
            if getattr(self, 'num_codebooks', None) and self.max_len < int(self.num_codebooks):
                logging.warning('TIGER: args.max_len (%d) < num_codebooks (%d). Adjusting max_len to num_codebooks.',
                                self.max_len, self.num_codebooks)
                self.max_len = int(self.num_codebooks)
                self.args.max_len = int(self.num_codebooks)
        except Exception:
            pass

        # Try to ensure vocab covers item ids if we fallback to item-id tokens
        try:
            n_items = int(getattr(self.corpus, 'n_items', 0))
        except Exception:
            n_items = 0
        if n_items and getattr(self.args, 'vocab_size', 0) < n_items + 1:
            self.args.vocab_size = n_items + 1

        # Define model params (may require transformers; errors will be logged)
        try:
            self._define_params()
            self.apply(self.init_weights)
        except Exception as e:
            logging.warning('TIGER: _define_params failed or transformers not available: %s', e)
            self.model = None

    def forward(self, feed_dict: dict) -> dict:
        if self.model is None:
            raise RuntimeError('TIGER model not initialized (transformers missing)')

        input_ids = feed_dict['history']
        attention_mask = feed_dict.get('attention_mask', None)
        labels = feed_dict.get('target', None)

        # NOTE:
        # We still pass `labels` so HF will create decoder_input_ids via shift_right(labels),
        # but we DO NOT use HF's returned loss because it ignores only label==-100; while
        # our padding uses pad_token_id. We compute loss ourselves in `loss()`.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        out = {
            'prediction': outputs.logits,
            'labels': labels,
            'hf_loss': None,  # force Route to our custom loss (pad-aware + optional label smoothing)
        }
        return out

    def loss(self, out_dict: dict) -> torch.Tensor:
        logits = out_dict['prediction']
        labels = out_dict['labels']
        if labels is None:
            raise ValueError('TIGER.loss requires decoder labels in out_dict["labels"]')

        pad = self.pad_token_id
        eps = float(getattr(self.args, 'label_smoothing', 0.0))
        if eps < 0.0:
            eps = 0.0
        if eps > 0.5:
            # keep it sane; typical choices are 0.05~0.2
            eps = 0.5

        # logits: (B, T, V) or (N, V), labels: (B, T) or (N,)
        if logits.dim() == 2:
            # (N, V)
            if eps == 0.0:
                return F.cross_entropy(logits, labels, ignore_index=pad)
            log_probs = F.log_softmax(logits, dim=-1)                         # (N, V)
            nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (N,)
            smooth = -log_probs.mean(dim=-1)                                  # (N,)
            mask = (labels != pad)
            return ((1.0 - eps) * nll + eps * smooth)[mask].mean()

        # (B, T, V)
        vocab_size = logits.shape[-1]
        flat_logits = logits.view(-1, vocab_size)  # (B*T, V)
        flat_labels = labels.view(-1)              # (B*T,)

        if eps == 0.0:
            return F.cross_entropy(flat_logits, flat_labels, ignore_index=pad)

        log_probs = F.log_softmax(flat_logits, dim=-1)  # (B*T, V)
        nll = -log_probs.gather(dim=-1, index=flat_labels.unsqueeze(-1)).squeeze(-1)  # (B*T,)
        smooth = -log_probs.mean(dim=-1)  # (B*T,)
        mask = (flat_labels != pad)
        loss = ((1.0 - eps) * nll + eps * smooth)[mask].mean()
        return loss

    def inference(self, feed_dict: dict, num_beams: int = None) -> dict:
        if self.model is None:
            raise RuntimeError('TIGER model not initialized (transformers missing)')
        num_beams = self.beam_size if num_beams is None else num_beams
        input_ids = feed_dict['history']
        attention_mask = feed_dict.get('attention_mask', None)

        # If candidate item list is provided (training/eval in ReChorus), score each candidate
        # Expected feed_dict['item_id'] shape: (batch_size, n_candidates)
        if 'item_id' in feed_dict:
            # Candidate scoring mode: compute log-probability of each candidate sequence given history
            item_ids = feed_dict['item_id']

            # convert numpy arrays to torch if needed
            if not torch.is_tensor(input_ids):
                input_ids = torch.tensor(input_ids, device=self.model.device)
            if attention_mask is not None and not torch.is_tensor(attention_mask):
                attention_mask = torch.tensor(attention_mask, device=self.model.device)
            if not torch.is_tensor(item_ids):
                item_ids = torch.tensor(item_ids, device=self.model.device)

            batch_size = input_ids.shape[0]
            n_cands = item_ids.shape[1]

            # Build labels (decoder target token ids) for all candidates
            all_codes = []
            max_len_tgt = 0
            for b in range(batch_size):
                for k in range(n_cands):
                    try:
                        iid = int(item_ids[b, k].item())
                    except Exception:
                        iid = int(item_ids[b, k])
                    # get code mapping from corpus if available, else treat item id as single token
                    try:
                        code = self.corpus.item_codes[iid]
                    except Exception:
                        code = [iid]
                    code_list = [int(x) for x in code] if hasattr(code, '__iter__') and not isinstance(code, (str, bytes)) else [int(code)]
                    all_codes.append(code_list)
                    if len(code_list) > max_len_tgt:
                        max_len_tgt = len(code_list)

            # pad target codes
            pad = self.pad_token_id
            labels = torch.full((batch_size * n_cands, max_len_tgt), pad, dtype=torch.long, device=self.model.device)
            for idx, code in enumerate(all_codes):
                labels[idx, :len(code)] = torch.tensor(code, dtype=torch.long, device=self.model.device)

            # repeat history for each candidate
            seq_in_len = input_ids.shape[1]
            hist_rep = input_ids.unsqueeze(1).expand(-1, n_cands, -1).contiguous().view(batch_size * n_cands, seq_in_len)
            if attention_mask is not None:
                attn_rep = attention_mask.unsqueeze(1).expand(-1, n_cands, -1).contiguous().view(batch_size * n_cands, seq_in_len)
            else:
                attn_rep = None

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=hist_rep, attention_mask=attn_rep, labels=labels)
                logits = outputs.logits  # (batch*n_cands, tgt_len, vocab)

                # compute token-wise log-probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (batch*n_cands, tgt_len)

                # mask padding tokens
                mask = (labels != pad).long()
                token_log_probs = token_log_probs * mask
                seq_log_prob = token_log_probs.sum(dim=1)

            scores = seq_log_prob.view(batch_size, n_cands)
            return {'prediction': scores}

        # default: return beam generation sequences
        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_len,
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
        batch_size = input_ids.shape[0]
        gen = gen.view(batch_size, num_beams, -1)
        return {'prediction': gen}

    def _define_params(self):
        from transformers import T5ForConditionalGeneration, T5Config
        args = self.args

        # If item_codes are present in corpus, enlarge vocab if required
        try:
            if hasattr(self.corpus, 'item_codes') and getattr(self.corpus, 'item_codes', None) is not None:
                all_codes = self.corpus.item_codes
                max_tok = -1
                for c in all_codes:
                    try:
                        if hasattr(c, '__iter__') and not isinstance(c, (str, bytes)):
                            mc = max([int(x) for x in c]) if len(c) > 0 else -1
                        else:
                            mc = int(c)
                        if mc > max_tok:
                            max_tok = mc
                    except Exception:
                        continue
                if max_tok >= 0 and getattr(args, 'vocab_size', 0) <= max_tok:
                    args.vocab_size = int(max_tok + 1)
        except Exception:
            pass

        t5config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_heads=args.num_heads,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            vocab_size=args.vocab_size,
            pad_token_id=args.pad_token_id,
            eos_token_id=args.eos_token_id,
            decoder_start_token_id=args.pad_token_id,
            feed_forward_proj=args.feed_forward_proj,
        )
        self.model = T5ForConditionalGeneration(t5config)

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index: int) -> dict:
            # base feed contains 'user_id', 'item_id', and for sequential models 'history_items'
            feed = super()._get_feed_dict(index)
            model = self.model
            corpus = self.corpus

            # Lazy-load item_codes mapping: priority
            # 1) model.item_codes if set
            # 2) if args.tiger_code_path provided, load npy and set model.item_codes
            # 3) fallback: treat item_id as token id
            item_codes = getattr(model, 'item_codes', None)
            if item_codes is None:
                item_codes = getattr(corpus, 'item_codes', None)

            if item_codes is None and getattr(model, 'args', None) is not None:
                code_path = getattr(model.args, 'tiger_code_path', '')
                if code_path:
                    try:
                        import numpy as _np
                        loaded = _np.load(code_path, allow_pickle=True)

                        norm = []
                        try:
                            arr = _np.array(loaded)
                            if arr.ndim == 2:
                                for row in arr.tolist():
                                    norm.append(list(row))
                            elif arr.ndim == 1:
                                for v in arr.tolist():
                                    norm.append([int(v)])
                            else:
                                for v in loaded:
                                    norm.append(list(v) if hasattr(v, '__iter__') else [int(v)])
                        except Exception:
                            for v in loaded:
                                norm.append(list(v) if hasattr(v, '__iter__') else [int(v)])

                        try:
                            mv = max([max(r) if len(r) > 0 else -1 for r in norm]) if len(norm) > 0 else -1
                        except Exception:
                            mv = -1

                        cb_k = getattr(model, 'codebook_k', getattr(self, 'codebook_k', 256))
                        num_cb = getattr(model, 'num_codebooks', getattr(self, 'num_codebooks', 4))

                        if mv >= 0 and mv < cb_k:
                            logging.info('TIGER.Dataset: converting per-codebook indices to global token ids (codebook_k=%d)', cb_k)
                            converted = []
                            for row in norm:
                                codes = list(row)[:num_cb]
                                if len(codes) < num_cb:
                                    codes = codes + [0] * (num_cb - len(codes))
                                offsets = [int(c) + i * cb_k + 1 for i, c in enumerate(codes)]
                                converted.append(offsets)
                            model.item_codes = converted
                            item_codes = converted
                        else:
                            model.item_codes = norm
                            item_codes = norm

                    except Exception as e:
                        logging.warning('TIGER.Dataset: failed to load tiger_code_path %s: %s', code_path, e)
                        item_codes = None

            # history items -> token ids
            hist_items = feed.get('history_items', np.array([])).tolist()
            history_tokens = []
            for iid in hist_items:
                try:
                    if item_codes is not None:
                        code = item_codes[int(iid)]
                    else:
                        code = int(iid)
                except Exception:
                    code = int(iid)
                if hasattr(code, '__iter__') and not isinstance(code, (str, bytes)):
                    history_tokens.extend([int(x) for x in code])
                else:
                    history_tokens.append(int(code))

            # truncate/pad to max_len (left pad)
            history_tokens = history_tokens[-model.max_len:]
            pad_len = model.max_len - len(history_tokens)
            if pad_len > 0:
                history_tokens = [model.pad_token_id] * pad_len + history_tokens

            # target: first column of item_id is the positive target item
            target_item = int(feed['item_id'][0])
            try:
                if item_codes is not None:
                    tgt_code = item_codes[target_item]
                else:
                    tgt_code = [target_item]
            except Exception:
                tgt_code = [target_item]

            if hasattr(tgt_code, '__iter__') and not isinstance(tgt_code, (str, bytes)):
                target_tokens = [int(x) for x in tgt_code]
            else:
                target_tokens = [int(tgt_code)]

            feed_out = {
                'user_id': feed['user_id'],
                'item_id': feed['item_id'],
                'history': np.array(history_tokens, dtype=np.int64),
                'attention_mask': np.array([1 if t != model.pad_token_id else 0 for t in history_tokens], dtype=np.int64),
                'target': np.array(target_tokens, dtype=np.int64),
            }
            return feed_out
