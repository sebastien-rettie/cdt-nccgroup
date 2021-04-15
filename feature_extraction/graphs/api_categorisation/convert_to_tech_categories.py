"""
Convert windows SDK and windows driver doc api cateogries to use the technology instead of the header file
"""

import argparse, yaml, json, collections, os


def main(allAPI, target_dir, out_name):
    with open(allAPI, 'r') as f:
            apis = yaml.load(f, Loader=yaml.FullLoader)        
    tech_categories = {}
    unique_headers_seen = collections.Counter()

    for path, subdirs, files in os.walk(target_dir):
        for name in files:
            if name == "config.json":
                with open(os.path.join(path, name), 'r') as f:
                    config = json.load(f)
                headers = config["headers"]
                funcs = []

                for header in headers:
                    if header.endswith('.h'):
                        header = header[:-2]
                        
                    if header in apis.keys():
                        unique_headers_seen[header] += 1
                        for func in apis[header]:
                            funcs.append(func)

                if funcs:
                    tech_categories[config["tech"] + "_sdk"] = funcs # remove "_sdk" if not passing win SDK

    with open('win32API_categories_functionheaders.yaml', 'r') as f:
        winAPI = yaml.load(f, Loader=yaml.FullLoader)

    print(set(unique_headers_seen.keys()) ^ set(winAPI.keys()))
    # print(len(unique_headers_seen.keys()))
    # l = list(unique_headers_seen.keys())
    # print(sorted(l))

    with open(out_name, 'w') as f:
        yaml.dump(tech_categories, f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert header categories to tech categories")
    parser.add_argument("API_categorisation")
    parser.add_argument("target_dir")
    parser.add_argument("output_name")
    args = parser.parse_args()
    
    return (args.API_categorisation, args.target_dir, args.output_name)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
