#!/usr/bin/env python3
"""
Entry point cho pipeline xử lý dữ liệu
"""

import argparse
from pathlib import Path
from data_processor import CUB200Processor


def parse_args():
    parser = argparse.ArgumentParser(description='Process CUB-200 bird dataset')
    
    parser.add_argument('--cub-root', type=str, required=True,
                       help='Path to extracted CUB_200_2011 folder')
    
    parser.add_argument('--project-root', type=str, default='.',
                       help='Root directory of bird-search-system project')
    
    parser.add_argument('--target', type=int, default=500,
                       help='Target number of processed images (default: 500)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize processor
    processor = CUB200Processor(project_root=args.project_root)
    
    # Run pipeline
    stats = processor.run_pipeline(
        cub_root=args.cub_root,
        target_count=args.target
    )
    
    # Exit with appropriate code
    if stats['success'] >= args.target:
        print("\n🎉 Pipeline completed successfully!")
        return 0
    else:
        print("\n⚠️  Pipeline completed but target not reached.")
        print("   Check logs and consider adjusting filter criteria.")
        return 1


if __name__ == '__main__':
    exit(main())