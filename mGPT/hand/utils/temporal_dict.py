from mGPT.hand.utils.xdict import xdict
import pandas as pd
import torch
import numpy as np
class temporal_dict(xdict):

    def __init__(self, mydict=None):
        """
        Constructor for the xdict class. Creates a new xdict object and optionally initializes it with key-value pairs from the provided dictionary mydict. If mydict is not provided, an empty xdict is created.
        """
        if mydict is None:
            return

        for k, v in mydict.items():
            super().__setitem__(k, [v])
    
    def __setitem__(self, key, val):
        """
        Adds a new key-value pair to the xdict. If the key already exists, the value is appended to the existing list of values. If the key does not exist, a new key-value pair is added to the dictionary.
        """
        if key in self:
            self[key].append(val)
        else:
            super().__setitem__(key, [val])


    def add_dict(self, input_dict, tag=""):
        """
        Adds all key-value pairs from the provided dictionary input_dict to the xdict.
        """
        for k, v in input_dict.items():

            if torch.is_tensor(v):
                v = v.item()
            self.__setitem__(tag+str(k), v)

    def average(self):
        """
        Returns the average of all values in the xdict.
        """
        avg_dict = xdict()
        for k, v in self.items():
            
            if type(v[0]) == str:
                continue

            count = len(v)
            total = sum(v)
            
            ## added this to help json serialize later
            if torch.is_tensor(total):
                total =  total.item()

            avg_dict[k] = total / count 

        return avg_dict
    
    def dumps_as_csv(self, out_file):
        """
        Dumps the xdict as a JSON file with the provided filename.
        """

        ### seq_wise_dict
        # seq_wise_dict = { k:v for k,v in self.items() }
        seq_wise_dict = { k:v for k,v in self.items() if len(v) > 1}
        df = pd.DataFrame(seq_wise_dict)
        df.to_csv(out_file, index=False)
        print("Dumped seqwise file out", out_file)

        # ### dump the avg dict
        # avg_dict = self.average()
        # df = pd.DataFrame(avg_dict, index=[0])
        # df.to_csv(out_file, index=False)
        # print("Dumped seqwise metric file", out_file)

    def compute_mean_conf_dict(self, replication_times=3):

        mean_conf_dict = {}
        for key, item in self.items(): 
            mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
            mean_conf_dict[key + "/mean"] = mean
            mean_conf_dict[key + "/conf_interval"] = conf_interval
        
        return mean_conf_dict
    

    def dump_mean_conf_dict(self, outfile, replication_times=3, out_metric=None):

        if out_metric is None:
            out_metric = self.compute_mean_conf_dict(replication_times)

        import json
        out_metric_ = {}
        for k, v in out_metric.items():
            
            if torch.is_tensor(v):
                out_metric_[k] = v.item()
            else:
                out_metric_[k] = float(v)

        # outfile = os.path.join(output_dir, "metric.json")
        with open(outfile, "w") as f:
            json.dump(out_metric_, f, indent=4)
        print(f"Metric dumped at: {outfile}")

        df = pd.DataFrame(out_metric_, index=[0])
        df.to_csv(outfile.replace(".json", ".csv"), index=False)
        print("Dumped csv file", outfile.replace(".json", ".csv"))



def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval
