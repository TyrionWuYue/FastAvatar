import json
from functools import partial
import torch

__all__ = ['MixerDataset']


class MixerDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 split: str,
                 subsets: list,
                 **dataset_kwargs,
                 ):
        # Handle both old format (meta_path[split]) and new format (meta_path as string)
        filtered_subsets = []
        for subset in subsets:
            meta_path = subset.get("meta_path")
            if isinstance(meta_path, dict):
                # Old format: meta_path is a dict with 'train' and 'val' keys
                if meta_path.get(split) is not None:
                    filtered_subsets.append(subset)
            elif isinstance(meta_path, str):
                # New format: meta_path is a string, use it for both train and val
                filtered_subsets.append(subset)
            else:
                # Skip if meta_path is None
                continue
        
        # Load and filter data for each subset in MixerDataset (centralized filtering)
        # This avoids needing filtering logic in individual dataset classes
        # Optimize: if multiple subsets share the same meta_path, load JSON only once
        meta_path_cache = {}
        self.subsets = []
        for subset in filtered_subsets:
            # Get the meta_path for this subset
            meta_path = subset.get("meta_path")
            if isinstance(meta_path, dict):
                meta_path = meta_path[split]
            
            # Load JSON only once per unique meta_path
            if meta_path not in meta_path_cache:
                with open(meta_path, 'r') as f:
                    meta_path_cache[meta_path] = json.load(f)
            
            # Filter data based on dataset name
            filtered_data = self._filter_data_for_subset(
                meta_path_cache[meta_path], 
                subset['name']
            )
            
            # Create dataset with pre-filtered data (pass dict instead of file path)
            dataset_fn = self._dataset_fn(subset, split)
            # Pass val_id from subset config if available
            subset_kwargs = {**dataset_kwargs}
            if 'val_id' in subset:
                subset_kwargs['val_id'] = subset['val_id']
            self.subsets.append(dataset_fn(meta_path=filtered_data, **subset_kwargs))
        
        self.virtual_lens = [
            len(subset_obj)
            for subset_config, subset_obj in zip(filtered_subsets, self.subsets)
        ]

    @staticmethod
    def _filter_data_for_subset(all_frame_groups: dict, dataset_name: str):
        """Filter data based on dataset name.
        
        Args:
            all_frame_groups: Dictionary of all frame groups from JSON
            dataset_name: Name of the dataset ("nersemble" or "vfhq")
        
        Returns:
            Filtered dictionary with prefix STRIPPED.
        """
        filtered = {}
        prefix = f"{dataset_name}/"
        
        count_prefix = 0
        count_legacy = 0
        
        for path, frame_data in all_frame_groups.items():
            # Strategy 1: Check Prefix (New Format)
            if path.startswith(prefix):
                # Clean key: "nersemble/p1/01" -> "p1/01"
                # This is CRITICAL because sub-datasets expect clean paths
                clean_key = path[len(prefix):]
                filtered[clean_key] = frame_data
                count_prefix += 1
                continue
                
            # Strategy 2: Heuristic Fallback (Legacy)
            # Only use if NOT matching other prefixes (to avoid double match if possible)
            if "/" in path and path.split('/')[0] not in ["nersemble", "vfhq"]: 
                # ^ Simple check to avoid checking keys that are meant for other datasets but just don't have this dataset's prefix
                
                # VFHQ Fallback
                if dataset_name == "vfhq":
                    path_parts = path.split('/')
                    is_vfhq = False
                    if len(path_parts) >= 2 and "EXP-" in path_parts[1]:
                         is_vfhq = True
                    elif len(path_parts) == 2 and path_parts[0].isdigit() and "EXP-" in path:
                         is_vfhq = True
                    
                    if is_vfhq:
                        filtered[path] = frame_data
                        count_legacy += 1
                
                # Nersemble Fallback
                elif dataset_name == "nersemble":
                     # Inverse of VFHQ check, essentially
                     # This is risky, but kept for strict backward compatibility if needed
                     pass

        print(f"Filtered {len(filtered)} items for '{dataset_name}' (Prefix: {count_prefix}, Legacy: {count_legacy})")
        return filtered

    @staticmethod
    def _dataset_fn(subset_config: dict, split: str):
        name = subset_config['name']

        dataset_cls = None
        if name == "nersemble":
            from .nersemble import NersembleDataset
            dataset_cls = NersembleDataset
        elif name == "vfhq":
            from .vfhq import VFHQDataset
            dataset_cls = VFHQDataset
        else:
            raise NotImplementedError(f"Dataset {name} not implemented")
        print("==="*16*3, "\nUse dataset loader:", name, "\n"+"==="*3*16)

        # Handle both old and new meta_path formats
        # Note: meta_path will be replaced with filtered dict in __init__
        meta_path = subset_config.get("meta_path")
        if isinstance(meta_path, dict):
            meta_path = meta_path[split]

        return partial(
            dataset_cls,
            root_dirs=subset_config['root_dirs'],
            meta_path=meta_path,  # Will be replaced with filtered dict in __init__
        )
    

    def __len__(self):
        return sum(self.virtual_lens)

    def __getitem__(self, idx):
        subset_idx = 0
        virtual_idx = idx
        while virtual_idx >= self.virtual_lens[subset_idx]:
            virtual_idx -= self.virtual_lens[subset_idx]
            subset_idx += 1
        real_idx = virtual_idx % len(self.subsets[subset_idx])
        return self.subsets[subset_idx][real_idx]
