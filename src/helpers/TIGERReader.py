# -*- coding: UTF-8 -*-
"""
TIGERReader

Specialized reader to make TIGER work with ReChorus datasets and possibly
imperfect `.npy` code files (e.g. Grocery_and_Gourmet_Food outputs).

Responsibilities:
- Load train/dev/test as SeqReader does (inherit from SeqReader).
- Load / discover `tiger_code_path` .npy files, normalize them into a
  list-of-lists of global token ids (per-codebook -> offset conversion),
  handle 1-based arrays, and attach the result as `self.item_codes`.
- Optionally save a converted copy to the dataset folder for inspection.
"""
import os
import glob
import logging
import numpy as np

from helpers.SeqReader import SeqReader


class TIGERReader(SeqReader):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--tiger_code_path', type=str, default='',
                            help='(Optional) npy file mapping item_id -> code arrays. If empty, try to discover in dataset folder.')
        parser.add_argument('--codebook_k', type=int, default=256,
                            help='Entries per codebook (for tiger codes)')
        parser.add_argument('--num_codebooks', type=int, default=4,
                            help='Number of codebooks per item')
        parser.add_argument('--save_converted_codes', type=int, default=1,
                            help='If set, save converted codes to dataset folder')
        return SeqReader.parse_data_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # after SeqReader init, we have self.n_items, self.data_df etc.
        self.item_codes = None
        self._prepare_item_codes()

    def _discover_code_path(self):
        # if explicit path provided, use it (absolute or relative to data/dataset)
        path = getattr(self.args, 'tiger_code_path', '')
        if path:
            if os.path.isabs(path) and os.path.exists(path):
                return path
            # try relative to dataset folder
            cand = os.path.join(self.prefix, self.dataset, path)
            if os.path.exists(cand):
                return cand
            # try relative to project
            if os.path.exists(path):
                return path

        # otherwise try to discover common filenames under dataset folder
        ds_folder = os.path.join(self.prefix, self.dataset)
        if os.path.isdir(ds_folder):
            patterns = ['*t5_rqvae*.npy', '*item_codes*.npy', '*codes*.npy']
            for p in patterns:
                found = glob.glob(os.path.join(ds_folder, p))
                if found:
                    return found[0]
        return ''

    def _prepare_item_codes(self):
        code_path = self._discover_code_path()
        cb_k = getattr(self.args, 'codebook_k', 256)
        num_cb = getattr(self.args, 'num_codebooks', 4)
        save_conv = bool(getattr(self.args, 'save_converted_codes', 1))

        if code_path:
            try:
                loaded = np.load(code_path, allow_pickle=True)
                logging.info('TIGERReader: loaded codes from %s (shape=%s)', code_path, getattr(loaded, 'shape', None))
                norm = []
                try:
                    arr = np.array(loaded)
                    if arr.ndim == 2:
                        for row in arr.tolist():
                            norm.append(list(row))
                    elif arr.ndim == 1:
                        # could be array of objects (lists) or simple ints
                        for v in arr.tolist():
                            if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                                norm.append(list(v))
                            else:
                                norm.append([int(v)])
                    else:
                        for v in loaded:
                            norm.append(list(v) if hasattr(v, '__iter__') else [int(v)])
                except Exception:
                    for v in loaded:
                        norm.append(list(v) if hasattr(v, '__iter__') else [int(v)])

                # detect per-codebook small indices
                try:
                    mv = max([max(r) if len(r) > 0 else -1 for r in norm]) if len(norm) > 0 else -1
                except Exception:
                    mv = -1

                converted = None
                if mv >= 0 and mv < cb_k:
                    logging.info('TIGERReader: converting per-codebook indices to global token ids (codebook_k=%d,num_codebooks=%d)', cb_k, num_cb)
                    converted = []
                    for row in norm:
                        codes = list(row)[:num_cb]
                        if len(codes) < num_cb:
                            codes = codes + [0] * (num_cb - len(codes))
                        offsets = [int(c) + i * cb_k + 1 for i, c in enumerate(codes)]
                        converted.append(offsets)
                else:
                    # assume already global token ids
                    converted = [list(r) if hasattr(r, '__iter__') and not isinstance(r, (str, bytes)) else [int(r)] for r in norm]

                # if loaded length equals n_items+1, drop first index (1-based -> 0-based)
                try:
                    if hasattr(self, 'n_items') and self.n_items and len(converted) == self.n_items + 1:
                        logging.info('TIGERReader: detected 1-based codes array; dropping index 0 to align with 0-based item ids')
                        converted = converted[1:]
                except Exception:
                    pass

                self.item_codes = converted
                # attach to reader so model.corpus.item_codes is available
                setattr(self, 'item_codes', self.item_codes)

                if save_conv:
                    try:
                        out_path = os.path.join(self.prefix, self.dataset, 'item_codes_converted.npy')
                        np.save(out_path, np.array(self.item_codes, dtype=object))
                        logging.info('TIGERReader: saved converted codes to %s', out_path)
                    except Exception as e:
                        logging.warning('TIGERReader: failed to save converted codes: %s', e)

            except Exception as e:
                logging.warning('TIGERReader: failed to load code_path %s: %s', code_path, e)
                self.item_codes = None
        else:
            # no code file: build deterministic mapping so TIGER can run
            try:
                n_items = int(getattr(self, 'n_items', 0))
            except Exception:
                n_items = 0
            if n_items and n_items > 0:
                logging.info('TIGERReader: no code file found; building deterministic codes for %d items', n_items)
                built = []
                for iid in range(n_items):
                    codes = [int((iid + i) % cb_k) for i in range(num_cb)]
                    offsets = [c + i * cb_k + 1 for i, c in enumerate(codes)]
                    built.append(offsets)
                self.item_codes = built
                setattr(self, 'item_codes', self.item_codes)
                if save_conv:
                    try:
                        out_path = os.path.join(self.prefix, self.dataset, 'item_codes_generated.npy')
                        np.save(out_path, np.array(self.item_codes, dtype=object))
                        logging.info('TIGERReader: saved generated codes to %s', out_path)
                    except Exception as e:
                        logging.warning('TIGERReader: failed to save generated codes: %s', e)
            else:
                logging.info('TIGERReader: cannot build item_codes (n_items unknown)')

