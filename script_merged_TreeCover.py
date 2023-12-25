# importing libraries
import pandas as pd
import glob
import os
import re
import wikipedia as wiki
# # merging the files
# joined_files = os.path.join(os.path.dirname(__file__), "1_*.csv")

# # A list of all joined files is returned
# joined_list = glob.glob(joined_files)

# # # Finally, the files are joined
# df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
# df.insert(0, 'index', range(len(df)), allow_duplicates=False)

# print(df)

def main():
    dic_Borough = {
    'Manhattan': 100,
    'Bronx': 200,
    'Brooklyn': 300,
    'Queens': 400,
    'Staten Island': 500
    }

    
    taxi_zone_lookup = pd.read_csv('/home/featurize/data/target_usi/taxi_zone_lookup.csv')
    treecover = pd.read_csv(('/home/featurize/data/target_usi/treecover.csv'),index_col=0)
    
    for index, row in taxi_zone_lookup.iterrows():
        lookup_str = row['Borough'] + ' ' + row['Zone']
        if row['Borough'] == 'EWR' or row['Borough'] == 'Unknown':
            continue
        return_rearch_code = -1
        try:
            page_ = wiki.page(lookup_str)
        except Exception:
            print(lookup_str+"  exception")
            continue
        else:
            print(lookup_str)
            C_begin_number = page_.content.find('Community District')
            if C_begin_number != -1:
                # 说明有            
                return_rearch_code = int((re.findall(r'\d+\.\d+|\d+',page_.content[C_begin_number+18:C_begin_number+22])[0]))


            if return_rearch_code == -1:
                continue
            row_index = dic_Borough[row['Borough']]+return_rearch_code
            col_index = 'TreeCover'
            taxi_zone_lookup.at[index, 'TreeCover'] = treecover.at[row_index,col_index]

    taxi_zone_lookup.to_csv('/home/featurize/data/target_usi/merged_TreeCover.csv', index=False)   

if __name__ == "__main__":
    main()

