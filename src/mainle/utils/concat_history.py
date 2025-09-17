import json
import argparse


# load json files and concatenate them into a single json
def concat_json_files(json_files, output_file, print_messages=False):
    data = []
    for json_file in json_files:
        with open(json_file) as f:
            raw_data = json.load(f)
            data.extend(raw_data)

    if print_messages:
        for message in data:
            print(f"[{message['model']}] >> {message['content']}")

    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        "-f",
        help="Folder where json files are saved",
        default="../history/experiments/",
    )
    parser.add_argument(
        "--file_prefix",
        "-p",
        help="Prefix of the json files",
        type=str,
        default="999",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help="Output file name",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parse_args()

    path_prefix = f"{args.folder}{args.file_prefix}_"

    json_files = [
        f"{path_prefix}1_parser.json",
        f"{path_prefix}2_explainer.json",
        f"{path_prefix}3_simplifier.json",
        f"{path_prefix}4_comparison.json"
    ]

    if not args.output_file:
        args.output_file = f"{path_prefix}full_history.json"

    concat_json_files(json_files, args.output_file)

    print(f"Concatenated json files into {args.output_file}")
