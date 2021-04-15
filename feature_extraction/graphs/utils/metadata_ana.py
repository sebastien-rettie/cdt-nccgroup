import argparse, os, shutil, collections
import yaml
import numpy as np
from matplotlib import pyplot as plt


def main(target_dir):

    timestamps = []
    sizes_kb = []
    malware_types = []

    for i, data_file in enumerate(os.listdir(target_dir)):
        with open(os.path.join(target_dir, data_file), 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)["data"]

        if "PE32" not in data["filetype"]:
            continue

        try:
            timestamp = int(data["exif"]["TimeStamp"].split(":")[0])
            if timestamp != 0:
                timestamps.append(timestamp)
        except KeyError:
            pass

        try:
            size, unit = data["exif"]["FileSize"].split(' ')
            if unit == 'kB':
                sizes_kb.append(float(size))
            elif unit == 'MB':
                sizes_kb.append(float(size)/1000.0)
            else:
                print(data["exif"])
        except KeyError:
            pass

        votes = []
        for report in data["virustotal"]["scans"].values():
            if report["detected"] == False:
                continue
            string = report["result"]
            for delimeter in ["-", " ", "/", ":"]:
                string = string.replace(delimeter, ".")
            strings = string.split('.')
            votes.extend([ string for string in strings if string not in ["Trojan", "Troj", "malware", "Win32", "malicious",
                                                                          "Graftor", "Virus", "AIT", "Malware", "Win64", "Application",
                                                                          "Dell", ] ])
        vote = collections.Counter(votes).most_common(1)[0][0]
        if "Gen" in vote:
            vote = "Generic"
        malware_types.append(vote)

        if ((i + 1) % 100) == 0:
            print('[{}/{}] - ({:2.2%})'.format(i+1, len(os.listdir(target_dir)), float(i + 1)/float(len(os.listdir(target_dir)))))

    print(collections.Counter(timestamps))
    plt.rc('font', family='serif')
    timestamps_plot = [timestamp for timestamp in timestamps if timestamp <= 2020]
    plt.hist(timestamps_plot, bins=max(timestamps_plot) - min(timestamps_plot), histtype='step')
    plt.xlabel('year', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    malware_types = collections.Counter(malware_types)
    reduced_malware_types = collections.Counter({ key : cnt for key, cnt in malware_types.items() if cnt >= 100 })
    print(collections.Counter(reduced_malware_types))
    x = list(reduced_malware_types.values())
    y = list(reduced_malware_types.keys())
    x_coords= np.arange(len(reduced_malware_types.keys()))
    plt.bar(x_coords, x, align='center')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.xticks(x_coords, y)
    plt.xlabel('malware type', fontsize=18)
    plt.xticks(rotation=60, ha='center', fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    plt.hist(sizes_kb, bins=np.logspace(np.log10(1),np.log10(max(sizes_kb)), 50), histtype='step')
    plt.gca().set_xscale("log")
    plt.xlabel('size (kB)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


def parse_arguments():

    parser = argparse.ArgumentParser(description="Analyse VirusShare reports")
    parser.add_argument("target_dir")
    args = parser.parse_args()
    
    return (args.target_dir,)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)