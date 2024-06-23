import argparse
from app.store import S3Store
from app.constants import *
from app.train_imdb import run_all

def handle_store(args):
    print('\n(￣个￣) store bot here sir!\n')

    s3_store = S3Store(local_dir=args.local_dir)
    if args.action == 'download':
        s3_store.download(args.file_path)
    elif args.action == 'upload':
        s3_store.upload(args.file_path)
    elif args.action == 'overwrite_local':
        s3_store.overwrite_local(args.file_path)
    elif args.action == 'overwrite_remote':
        s3_store.overwrite_remote(args.file_path)
    elif args.action == 'sync':
        s3_store.sync()
    elif args.action == 'remote':
        s3_store.show_remote_files()
    elif args.action == 'create_bucket':
        s3_store.create_bucket_if_not_exists()
    else:
        print(f"Unknown action: {args.action}")

def handle_train(args):
    run_all()

    print('ʕノ•ᴥ•ʔノ\ntraining complete, hopefully nothing went horribly wrong...\n')

def main():
    parser = argparse.ArgumentParser(description='CLI for various tools.')
    
    subparsers = parser.add_subparsers(title='Tools', description='Available tools', dest='tool', required=True)

    # Store tool
    parser_store = subparsers.add_parser('store', help='S3 storage related actions')
    parser_store.add_argument('action', choices=['download', 'upload', 'overwrite_local', 'overwrite_remote', 'sync', 'remote', 'create_bucket'], help='Action to perform')
    parser_store.add_argument('--file_path', type=str, help='Path to the file for the action')
    parser_store.add_argument('--local_dir', type=str, default=LOCAL_DATA_PATH, help='Local directory path')
    parser_store.set_defaults(func=handle_store)

    # Train tool
    parser_train = subparsers.add_parser('train', help='Train the IMDB model')
    parser_train.set_defaults(func=handle_train)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()



