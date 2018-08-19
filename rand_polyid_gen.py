import sys
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://manager:toric@129.10.135.170/MLEARN")
MLEARN = client.MLEARN

n_samples = sys.argv[1]
file_path = sys.argv[2]

all_poly_curs = MLEARN.POLY.find({}, {"_id": 0, "POLYID": 1})
max_polyid = list(all_poly_curs.sort([("POLYID", -1)]).limit(1))[0]["POLYID"]

rand_polyids = np.random.randint(1, max_polyid, n_samples).tolist()
unmarked_rand_poly_curs = MLEARN.POLY.find({"POLYID": {"$in": rand_polyids}, "nfsrtMARK": {"$exists": False}}, {"_id": 0, "POLYID": 1})
unmarked_rand_polyid_iter = map(lambda x: x["POLYID"], rand_poly_curs.hint([("POLYID", 1)]).limit(n_samples))
unmarked_rand_polyids = list(unmarked_rand_polyid_iter)

print(str(unmarked_rand_polyids).replace(" ", ""), file = open(file_path, "w"), end = "")
print("{} Unmarked POLYIDs written to file {}.".format(len(unmarked_polyids), file_path)

client.close()