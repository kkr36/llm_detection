# enter: years, entrance file, prior csv (none == make a blank csv, path == append to the existing csv and save to new name? eg have an index 0 that keeps going up each time you pass it in)

# for each year:
    # for each alpha in {0 (no_drop), alpha_year}:
        # load in models, given path (both the pt file with pn and the pt file with pu)
        
        # for each model:
            # given path, parse out what experiments you want to rerun/ assemble test sets:
                # if combine, need to fix 2014-2020 (or whatever interval; most recently 2018-2020)
            # for each test set:
                # get: preds (save these in the same folder with year and alpha)
                # bbe (keep upper/lower conf bounds returned by function), 
                # avg pred pos / neg / avg(avg pos, avg neg) / avg(tpr, fpr) aka plugin (bootstrap 90% bounds?)
                # put into a new csv (train_year, train_method, train_alpha, test_alpha, test_year, **test_metrics); save/add to global df