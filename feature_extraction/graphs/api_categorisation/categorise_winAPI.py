"""
Clone github.com/MicrosoftDocs/windows-driver-docs-ddi then run on windows-driver-docs-ddi/wdk-ddi-src/content/.
"""

import os, argparse
import yaml

def search_directory(target_dir):
    categories = {}

    for path, subdirs, files in os.walk(target_dir):
        for name in files:
            if os.path.splitext(name)[1] == '.md':
                with open(os.path.join(path, name), 'r') as f:
                    get = False
                    api_names = []
                    for line in f.readlines():
                        if line.startswith('title:'):
                            title_line = line.strip('\n')

                        if 'api_name:' in line:
                            get = True

                        elif not line.startswith(' - '):
                            get = False

                        elif get:
                            api_names.append(line.strip('\n'))

                if 'function' not in title_line and 'macro' not in title_line:
                    continue
                
                for api_line in api_names:
                    api_name = api_line.split('-')[1].strip(' ')
                    categories.setdefault(os.path.basename(path), []).append(api_name)

    return categories
    

def main(target_dir):
    categories = search_directory(target_dir)

    with open('WDKAPI_categories_functionheaders.yaml', 'w') as f:
        yaml.dump(categories, f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse directory for functions and categorise them")
    parser.add_argument("target_dir")
    args = parser.parse_args()
    
    return (args.target_dir,)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
