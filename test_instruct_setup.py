#!/usr/bin/env python3
"""Test script to verify instruction tuning setup without running training."""

import torch
import sys
sys.path.append('.')

from model.processor import Processor
from model.vision import VisionConfig
from data.llava import LLaVAInstructDataset
from train.s2_qwen3v_instruct import make_instruct_collate_fn, VLInstructDataModule

def test_instruct_setup():
    print("Testing instruction tuning setup...")
    
    # 1. Create processor
    print("\n1. Creating processor...")
    try:
        vision_config = VisionConfig(
            n_embed=1280,
            n_layer=32,
            n_heads=16,
            output_n_embed=3584,
            in_channels=3,
            spatial_merge_size=2,
            spatial_patch_size=14,
            temporal_patch_size=2,
            intermediate_size=3420,
            hidden_act="silu",
        )
        processor = Processor(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct", 
            vision_config=vision_config
        )
        print("[OK] Processor created successfully")
    except Exception as e:
        print(f"[FAIL] Processor creation failed: {e}")
        return False
    
    # 2. Test dataset creation  
    print("\n2. Testing dataset creation...")
    try:
        dataset = LLaVAInstructDataset(cache_dir="./cache")
        print(f"[OK] Dataset created with {len(dataset)} samples")
        
        # Test a single sample
        sample = dataset[0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Messages: {len(sample['messages'])} turns")
        print(f"   First message: {sample['messages'][0]['role']}")
        
        # Print first few messages to understand format
        for i, msg in enumerate(sample['messages'][:2]):
            print(f"   Message {i}: {msg['role']} - {len(msg['content'])} content items")
            
    except Exception as e:
        print(f"[FAIL] Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Test collate function and sequence length analysis
    print("\n3. Testing collate function and analyzing sequence lengths...")
    try:
        collate_fn = make_instruct_collate_fn(processor, max_seq_len=1024)
        
        # Test with a small batch
        print("   Testing with first sample...")
        batch = [dataset[0]]
        collated = collate_fn(batch)
        
        if collated is None:
            print("[WARN] Collate function returned None (sequences too long)")
        else:
            print("[OK] Collate function working")
            print(f"   Input shape: {collated['input_ids'].shape}")
            print(f"   Labels shape: {collated['labels'].shape}")
            print(f"   Has pixels: {collated['pixels'] is not None}")
            
            # Check label masking
            labels = collated['labels']
            unmasked = (labels != -100).sum().item()
            total = labels.numel()
            print(f"   Unmasked tokens: {unmasked}/{total} ({unmasked/total*100:.1f}%)")
            
            # Show some token details
            input_ids = collated['input_ids'][0]
            labels_seq = labels[0]
            print(f"   Sequence length: {input_ids.shape[0]}")
            print(f"   Image pad tokens: {(input_ids == processor.image_pad_token_id).sum().item()}")
        
        # Analyze sequence lengths across dataset
        print(f"\n   Analyzing sequence lengths across entire dataset ({len(dataset)} samples)...")
        lengths = []
        skipped_count = 0
        
        from tqdm import tqdm
        for i in tqdm(range(len(dataset)), desc="Processing samples"):
            try:
                out = processor(messages=dataset[i]["messages"], device=None)
                seq_len = out["input_ids"].squeeze(0).numel()
                if seq_len <= 1024:
                    lengths.append(seq_len)
                else:
                    skipped_count += 1
            except Exception as e:
                skipped_count += 1
                continue
        
        if lengths:
            import numpy as np
            lengths = np.array(lengths)
            print(f"   Processed {len(lengths)} sequences (skipped {skipped_count})")
            print(f"   Min length: {lengths.min()}")
            print(f"   Max length: {lengths.max()}")
            print(f"   Mean length: {lengths.mean():.1f}")
            print(f"   Median length: {np.median(lengths):.1f}")
            print(f"   Sequences > 1024: {(lengths > 1024).sum()}")
            print(f"   Length percentiles: 50%={np.percentile(lengths, 50):.0f}, 90%={np.percentile(lengths, 90):.0f}, 95%={np.percentile(lengths, 95):.0f}")
            
    except Exception as e:
        print(f"[FAIL] Collate function failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test datamodule creation
    print("\n4. Testing datamodule creation...")
    try:
        dm = VLInstructDataModule(
            dataset=dataset,
            processor=processor,
            batch_size=1,  # Small batch for test
            max_seq_len=1024,
            num_workers=0,  # No multiprocessing for test
        )
        
        dataloader = dm.train_dataloader()
        print(f"[OK] DataModule created with {len(dataloader)} batches")
        
        # Test getting first batch
        print("   Getting first batch...")
        first_batch = next(iter(dataloader))
        if first_batch is not None:
            print(f"   First batch input_ids: {first_batch['input_ids'].shape}")
            print(f"   First batch labels: {first_batch['labels'].shape}")
            print(f"   First batch pixels: {first_batch['pixels'].shape if first_batch['pixels'] is not None else None}")
        else:
            print("   First batch is None (all sequences filtered out)")
            
    except Exception as e:
        print(f"[FAIL] DataModule creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nAll tests passed! Ready for instruction tuning.")
    return True

if __name__ == "__main__":
    success = test_instruct_setup()
    sys.exit(0 if success else 1)