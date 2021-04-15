"""
Clone github.com/MicrosoftDocs/cpp-docs then run on cpp-docs/docs/c-runtime-library/
"""

import os, argparse
import yaml

def search_directory(target_dir):
    categories = {}

    for path, subdirs, files in os.walk(target_dir):
        for name in files:
            if os.path.splitext(name)[1] == '.md':
                with open(os.path.join(path, name), 'r') as f:
                    get, getready = False, False
                    headers_table = []
                    for line in f.readlines():
                        if line.startswith('api_name:'):
                            api_line = line.strip('\n')

                        if line.startswith('helpviewer_keywords:'):
                            keywords_line = line.strip('\n')

                        if '## Requirements' in line:
                            getready = True
                        
                        elif getready and '|Routine' in line:
                            get = True

                        elif not line.startswith('|'):
                            get = False

                        elif get:
                            headers_table.append(line.strip('\n'))

                if 'function' not in keywords_line and 'macro' not in keywords_line:
                    continue

                if headers_table == []:
                    continue

                api_list = api_line.split('[')[1].strip(']')
                api_names = [ api_name.strip('"') for api_name in api_list.split(', ') ]
                missed_api_names = [ api_name for api_name in api_names if api_name not in ''.join(headers_table) ]
                seen_headers, seen_api_names = [], []

                for row in headers_table[1:]:
                    row_api_names = [ api_name for api_name in api_names if api_name in row ]

                    for api_name in missed_api_names:
                        original_api_name = api_name
                        
                        if api_name.startswith('_o_'):
                            api_name = api_name[3:]
                        elif api_name.startswith('_o'):
                            api_name = api_name[2:]
                        elif api_name.startswith('_'):
                            api_name = api_name[1:]

                        if api_name in row_api_names:
                            row_api_names.append(original_api_name)

                    seen_api_names.extend(row_api_names)

                    row = row.split('|')[2]
                    if "<br />" in row:
                        Cheaders = [ chunk for chunk in row.split('<br />') if 'C:' in chunk ][0]
                
                    else:
                        Cheaders = row

                    Cheaders = (Cheaders.split('<')[1:])
                    headers = [ header.split('>')[0] for header in Cheaders ]
                
                    seen_headers.extend(headers)

                    for header in headers:
                        for api_name in row_api_names:
                            categories.setdefault(header, []).append(api_name)

                for header in set(seen_headers):
                    for api_name in api_names:
                        if api_name not in set(seen_api_names):
                            categories.setdefault(header, []).append(api_name)

    return categories
    

def main(target_dir):
    categories = search_directory(target_dir)

    with open('CruntimelibAPI_categories_functionheaders.yaml', 'w') as f:
        yaml.dump(categories, f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse directory for functions and categorise them")
    parser.add_argument("target_dir")
    args = parser.parse_args()
    
    return (args.target_dir,)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
