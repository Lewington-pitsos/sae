import argparse
from app.store import S3Store
from app.constants import *
from app.train_imdb import run_all
from app.rp import *

def handle_store(args):
    print('\n(￣个￣)\nstoremaster at your service!\n')

    s3_store = S3Store(local_dir=args.local_dir)
    if args.action == 'download':
        s3_store.download(args.file_path)
    elif args.action == 'upload':
        s3_store.upload(args.file_path)
    elif args.action == 'overwrite_remote':
        s3_store.sync_remote()
    elif args.action == 'sync':
        s3_store.sync()
    elif args.action == 'remote':
        s3_store.show_remote_files()
    elif args.action == 'purge':
        s3_store.delete_all_remote()
    elif args.action == 'create_bucket':
        s3_store.create_bucket_if_not_exists()
    else:
        print(f"Unknown action: {args.action}")

    print('\n(￣个￣)\ntask complete!\n')


def handle_train(args):
    print('\n┏ʕ •ᴥ•ʔ┛\ntime to train\n')
    
    run_all()

    print('\nʕノ•ᴥ•ʔノ\ntraining complete\n')

def handle_runpod(args):

    print("\n°˖✧◝(⁰▿⁰)◜✧˖°\nlets run those pods\n")


    if args.action == 'deploy':
        deploy_pod()
    elif args.action == 'purge':
        print('purging all pods...\n')
        destroy_all_pods()
    elif args.action == 'show':
        show_pods()

    print("\n°˖✧◝(⁰▿⁰)◜✧˖°\n")

def main():
    parser = argparse.ArgumentParser(description='CLI for various tools.')
    
    subparsers = parser.add_subparsers(title='Tools', description='Available tools', dest='tool', required=True)

    # Store tool
    parser_store = subparsers.add_parser('store', help='S3 storage related actions')
    parser_store.add_argument('action', choices=['download', 'upload', 'overwrite_remote', 'sync', 'purge', 'remote', 'create_bucket'], help='Action to perform')
    parser_store.add_argument('--file_path', type=str, help='Path to the file for the action')
    parser_store.add_argument('--local_dir', type=str, default=LOCAL_DATA_PATH, help='Local directory path')
    parser_store.set_defaults(func=handle_store)

    # Train tool
    parser_train = subparsers.add_parser('train', help='Train the IMDB model')
    parser_train.set_defaults(func=handle_train)

    # Runpod tool
    parser_runpod = subparsers.add_parser('runpod', help='Create a Runpod')
    parser_runpod.add_argument('action', choices=['deploy', 'purge', 'show'], help='Action to perform')
    parser_runpod.set_defaults(func=handle_runpod)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()



