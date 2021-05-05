import argparse, os, shutil, collections
import yaml
import numpy as np


def main(metadata_dir, input_dirs, output_dir):
    graph_md5s = [ name.split('_')[1].split('.')[0] for input_dir in input_dirs.split(',') for name in os.listdir(input_dir) if name.startswith("VirusShare_") ]
    families = {}

    loop_len = len(os.listdir(metadata_dir))
    for idx, metadata_file in enumerate(os.listdir(metadata_dir)):
        if metadata_file.split('_')[0] in graph_md5s:
            with open(os.path.join(metadata_dir, metadata_file), 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)['data']

            votes = []
            for report in data["virustotal"]["scans"].values():
                if report["detected"] == False: continue

                string = report["result"]
                if type(string) != str: continue

                for delimeter in ["-", " ", "/", ":"]:
                    string = string.replace(delimeter, ".")
                strings = string.split('.')
                votes.extend([ string for string in strings if string not in [
                    "Trojan", "Troj", "malware", "Win32", "malicious", "Virus", "AIT", "Malware",
                    "Win64", "Application", "Dell", "Generic", "Agent", "Gen", "Variant", "GenericKD"] ])
        
            vote = collections.Counter(votes).most_common(1)[0][0]

            if vote not in families.keys(): families[vote] = []
            families[vote].append("VirusShare_{}.npz".format(metadata_file.split('_')[0]))
            
        if ((idx + 1) % 1000 == 0): print("{}/{}".format(idx, loop_len))

    print(">= 500")
    for family in families.keys():
        if len(families[family]) >= 500:
            print("{}: {}".format(family, len(families[family])))

    for family in list(families.keys()):
        if len(families[family]) < 500: del families[family]

    with open(os.path.join(output_dir, "malware_families.yaml"), 'w') as f:
        yaml.dump(families, f)

    family_labels = { family : idx for idx, family in enumerate(families.keys()) }

    with open(os.path.join(output_dir, "malware_family_labels.yaml"), 'w') as f:
        yaml.dump(family_labels, f)

    for family in families.keys():
        print("Transfering {}...".format(family))
        for filename in families[family]:
            for input_dir in input_dirs.split(','):
                if filename in os.listdir(input_dir):
                    data = np.load(os.path.join(input_dir, filename))
                    
                    label = np.array([family_labels[family]])

                    np.savez(os.path.join(output_dir, filename), E=data['E'], X=data['X'], family=label)

            
def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyse VirusShare reports")

    parser.add_argument("metadata_dir", help="Directory where all the metadata csv are")
    parser.add_argument("input_dirs", 
        help="Where the graphs are (comma separated dirs eg.'path/train,path/valid,path/test')")
    parser.add_argument("output_dir", help="Where the graphs go")

    args = parser.parse_args()
    
    return (args.metadata_dir, args.input_dirs, args.output_dir)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)