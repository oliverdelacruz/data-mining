########################################################################################################################
#Project: Data mining
#Authors: Oliver De La Cruz
#Date: 22/10/2016
#Description: MapReduce for identifying identity pairs
########################################################################################################################

def mapper(key, value):
    # key: None
    # value: one line of input file

    # import libraries
    import sys
    import random

    #local variables (parameters)
    word_ids = dict()
    num_id = 6
    num_rows_per_band = 32
    num_per_band = 32
    num_hashes = num_per_band * num_rows_per_band
    num_shingles = 8193
    prime_number = 4294967291 # max 32bits prime number
    prime_number = 18446744073709551557 # max 64bits prime number

    #Transform data
    value = value.split()
    video_id = value[0][num_id:]
    shingles = map(int, sorted(value[1:]))

    # Generate random sequence across distributed system
    random.seed(2)
    a_hash = [random.randrange(sys.maxint) for _ in xrange(0, num_hashes)]
    random.seed(7)
    b_hash = [random.randrange(sys.maxint) for _ in xrange(0, num_hashes)]

    # Generate the hash numbers
    def min_hash_fn(a, b, sig):
        hashes = [(((a * x) + b) % prime_number) % num_shingles for x in sig]
        return min(hashes)

    # Calls the function to produce the hash values
    def get_min_hash_row(sig):
        hashes = [min_hash_fn(a, b, sig) for a, b in zip(a_hash, b_hash)]
        return hashes

    # Splits the rows into bands
    def get_band(l, n):
        for i in xrange(0, len(l), n):
            yield tuple(l[i:i + n])

    # Calls the function to create the signature matrix
    min_hash_row = get_min_hash_row(shingles)

    # Calls the function to splits the rows into bands
    banded = get_band(min_hash_row, num_per_band)

    # Emit values to the map reducer (# hash function - standard library in python)
    for key, band in enumerate(banded):
        value = video_id + str(hash(band))
        yield key, value
    # Emit values to the map reducer (# hash function - standard library in python) - 2 option
    #for key, band in enumerate(banded):
        #key = str(key) + str(hash(band))
        #yield key, video_id


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key

    # load  modules
    from itertools import combinations

    # local variables
    num_digits_video = 9
    dict_video_hash ={}
    rev_multidict = {}

    #Create a dict of video ids and hash values
    for line in values:
        dict_video_hash[int(line[:num_digits_video])] = line[num_digits_video:]

    #Identify duplicates
    for key_video, value_hash in dict_video_hash.items():
        rev_multidict.setdefault(value_hash, set()).add(key_video)
    list_duplicates = [list(values) for key, values in rev_multidict.items() if len(values) > 1]
    for duplicates in list_duplicates:
        for video_id1, video_id2 in combinations(sorted(duplicates), 2):
            yield video_id1, video_id2

    #values.sort() - 2 option
    #for video_id1, video_id2 in combinations(values, 2):
        #yield video_id1, video_id2

