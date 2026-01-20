# load in model, given path

# given path, parse out what experiments you want to rerun/ assemble test sets:
    # if combine, need to fix 2014-2020 (or whatever interval; most recently 2018-2020)
# for each test set:
    # get: preds (save these in the same folder with year and alpha)
    # bbe (keep upper/lower conf bounds returned by function), 
    # avg pred pos / neg / avg(avg pos, avg neg) / avg(tpr, fpr) aka plugin (bootstrap 90% bounds?)
    # put into a new csv; save