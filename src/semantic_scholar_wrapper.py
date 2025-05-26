
import time
import string 
import random
import requests
import os

from multiprocessing import Pool
from itertools import chain
from more_itertools import sliced

from loguru import logger
from IPython.display import display, clear_output

import pandas as pd
import numpy as np



my_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

paper_fields = [
    'paperId', 'title', 'url', 'venue', 'publicationVenue', 'year', 'journal', 'isOpenAccess',
    'publicationTypes', 'publicationDate',
    'fieldsOfStudy',
    'abstract',
    'authors.authorId', 
    'externalIds',
    'citationStyles',
    'citationCount', 'influentialCitationCount', 'citations.paperId',
    'referenceCount', 'references.paperId'
]

class SS:

    def __init__(self, api_key=my_api_key, max_attempts=10, verbose=True):
        self.api_key = api_key
        self.max_attempts = max_attempts
        self.verbose=True


    # -------------------------------------------------------------------------
    # Get an individual item (paper or author)
    
    def get_item(self, item_type, item_id, fields=[], sleep=0):

        time.sleep(sleep)
        
        req = requests.get
        
        url = 'https://api.semanticscholar.org/graph/v1/{}/{}'.format(
            item_type, item_id
        )
        
        api_args = {
            'params':  {'fields':','.join(fields)},
            'headers': {'X-API-KEY': self.api_key}, 
        }

        return self.call_api(req, url, api_args)

    def get_paper(self, paper_id, fields=[], sleep=0):

        data = self.get_item(item_type='paper', item_id=paper_id, fields=fields, sleep=sleep)

        # Check if we receive valid data
        if 'paperId' in data:
            return data
        else:
            return np.nan

    def get_author(self, author_id, fields=[], sleep=0):

        data = self.get_item(item_type='author', item_id=author_id, fields=fields, sleep=sleep)

        # Check if we receive valid data
        if 'authorId' in data:
            return data
        else:
            return np.nan


    
    def get_citations(self, paper_id, offset=0, limit=100, fields=['paperId', 'isInfluential'], sleep=0):
        
        time.sleep(sleep)
        
        req = requests.get
        
        url = 'https://api.semanticscholar.org/graph/v1/paper/{}/citations'.format(paper_id)

        api_args = {
            'params':  {'fields':','.join(fields), 'offset': offset, 'limit': limit},
            'headers': {'X-API-KEY': self.api_key}, 
        }

        data =  self.call_api(req, url, api_args)
                
        return data

    
    def get_all_citations(self, paper_id, total_citations, limit=1000, fields=['paperId', 'isInfluential'], sleep=0):

        all_citations, all_influential_citations = [], []

        for offset in range(0, total_citations, limit):
            citation_data = self.get_citations(paper_id, offset=offset, limit=limit, fields=fields, sleep=sleep)

            if citation_data == False: break        # Probably exceeded max attempts
            if 'data' not in citation_data: break   # Probably an error due to missing paper id in SS.
                
            # Otherwise, we can iterate over the citations and compile them into 
            # a list of all cites and influential cites.
            for citing_paper in citation_data['data']:
                citing_paper_id = citing_paper['citingPaper']['paperId']
                is_influential = citing_paper['isInfluential']
                
                all_citations.append(citing_paper_id)
                if is_influential: 
                    all_influential_citations.append(citing_paper_id)

        # Finished trying to get citations, return what we have.
        return all_citations, all_influential_citations


    

    # -------------------------------------------------------------------------
    # Get a set of items (papers or authors)

    def get_items(self, item_type, item_ids, fields=[], sleep=0, batch_num=None):
        
        time.sleep(sleep)
        
        req = requests.post
        
        url = 'https://api.semanticscholar.org/graph/v1/{}/batch'.format(item_type)
        
        api_args = {
            'params':  {'fields':','.join(fields)},
            'headers': {'X-API-KEY': self.api_key}, 
            'json':    {'ids': item_ids},
        }
        
        return self.call_api(req, url, api_args, batch_num=batch_num)

    
    def get_papers(self, paper_ids, fields=[], sleep=0, batch_num=None):

        if len(paper_ids)>500: raise Exception('Too many paper ids (max 500)')

        papers = self.get_items(item_type='paper', item_ids=paper_ids, fields=fields, sleep=sleep, batch_num=batch_num)

        if papers:
            
            # Remove missing data
            checked_papers = [
                paper 
                for paper in papers 
                if (type(paper) is dict) and ('paperId' in paper)
            ]
    
            return checked_papers

        return []

    
    def get_authors(self, author_ids, fields=[], sleep=0, batch_num=None):

        if len(author_ids)>100: raise Exception('Too many author ids (max 100)')
            
        authors = self.get_items(item_type='author', item_ids=author_ids, fields=fields, sleep=sleep, batch_num=batch_num)

        if authors:
            
            # Remove missing data
            checked_authors = [
                author 
                for author in authors 
                if (type(author) is dict) and ('authorId' in author)
            ]
    
            return checked_authors

        return []


    # -------------------------------------------------------------------------
    # Get batches of items (papers or authors)
    def get_item_batches(self, item_getter, item_ids, fields, batch_size, pool_size, sleep=0):

        item_batches = list(sliced(item_ids, batch_size))

        with Pool(pool_size) as p:

            batch_params = [(item_batch, fields, sleep, batch_num) for batch_num, item_batch in enumerate(item_batches)]
            items = p.starmap(item_getter, batch_params)

        # Flatten the item lists
        items = list(chain.from_iterable(items))
        
        if self.verbose: logger.info('Found {} items.'.format(len(items)))
        
        return items     

    
    def get_papers_in_batches(self, paper_ids, fields, batch_size=500, pool_size=1, sleep=1):

        papers = self.get_item_batches(
            self.get_papers, 
            paper_ids, fields, 
            batch_size=batch_size, pool_size=pool_size, sleep=sleep
        )
        
        # Validate the papers and return the valid list.
        if papers:

            # Remove missing data
            checked_papers = [
                paper 
                for paper in papers 
                if (type(paper) is dict) and ('paperId' in paper)
            ]

            if self.verbose: logger.info('Found {} checked papers.'.format(len(checked_papers))) 
            
            return checked_papers

        return []

    def get_authors_in_batches(self, author_ids, fields, batch_size=100, pool_size=1, sleep=1):

        authors = self.get_item_batches(
            self.get_authors, 
            author_ids, fields, 
            batch_size=batch_size, pool_size=pool_size, sleep=sleep
        )
        
        # Validate the authors and return the valid list.
        if authors:

            # Remove missing data
            checked_authors = [
                author 
                for author in authors 
                if (type(author) is dict) and ('authorId' in author)
            ]

            if self.verbose: logger.info('Found {} checked authors.'.format(len(checked_authors))) 
            
            return checked_authors

        return []


    # -------------------------------------------------------------------------
    # Item search
    def bulk_item_search(self, item_type, query, fos='', year='', pub_types='', fields=[], venue=None, token=None, batch_num=None):

        req = requests.get
        
        url = 'https://api.semanticscholar.org/graph/v1/{}/search/bulk'.format(item_type)

        params = {'query': query, 'fields':','.join(fields), 'fieldsOfStudy':fos, 'year':year, 'publicationTypes':pub_types}

        if venue is not None:
            params['venue'] = venue

        # If there is a token provided then add it.
        if token: params['token'] = token
            
        api_args = {
            'params':  params,
            'headers': {'X-API-KEY': self.api_key}, 
        }

        return self.call_api(req, url, api_args, batch_num=batch_num)


    
    def bulk_paper_search(self, query, fos='', year='', pub_types='', fields=[], venue=None, token=None, sleep=0, max_results=None, batch_num=0):

        # Get the first page of results (up to 1000)
        results = self.bulk_item_search(
            item_type='paper', query=query, fos=fos, year=year, pub_types=pub_types, fields=fields, venue=venue, token='', batch_num=batch_num
        )

        # print(results)
        
        token = results['token']  # The next page token
        papers = results['data']  # The current results, if any.

        # Keep iterating through next pages of results...
        while token:

            time.sleep(sleep)

            batch_num += 1
            
            results = self.bulk_item_search(
                item_type='paper', query=query, fos=fos, year=year, pub_types=pub_types, fields=fields, venue=venue, token=token, batch_num=batch_num
            )

            token = results['token']  # The new next page token
            papers = papers + results['data']  # The new results, if any.

            # Stop if we have hit max results, if specified
            if max_results is not None:
                if len(papers)>max_results: break
        

        return papers

    

    
    def exact_title_search(self, title, sleep=0, batch_num=None):

        time.sleep(sleep)

        # Get up to the top 10 results from a bulk paper search.
        results = self.bulk_paper_search(query=title, sleep=sleep, batch_num=batch_num, max_results=10)
        
        # Some papers have Abstract in front of them.
        title = title.replace('Abstract:', '').replace('Preface:', '').strip()


        remove_punctuation = str.maketrans('', '', string.punctuation)
        clean_title = title.translate(remove_punctuation).lower().strip()

        for result in results:

            clean_result_title = result['title'].translate(remove_punctuation).lower().strip()
            
            if clean_title==clean_result_title: 
                if self.verbose: logger.info('Batch {}, found match for {}'.format(batch_num, title))

                # We need to return the actual title and the paperid because some searches will fail
                # and we need a way to know which paper ids go with which titles
                return title, result['paperId']
                
        
        return title, np.nan

    

    def exact_title_search_in_batches(self, titles, pool_size=1, sleep=1):

        with Pool(pool_size) as p:
            search_params = [(title, sleep, title_num) for title_num, title in enumerate(titles)]
            result_ids = p.starmap(self.exact_title_search, search_params)

        return [result_id for result_id in result_ids if result_id]
        
        

    # -------------------------------------------------------------------------
    # The main call function wraps the reques in a try so that if it fails 
    # we can catch the exception and try again.
    def call_api(self, req, url, api_args, batch_num=None):
        
        for attempt in range(self.max_attempts):

            try:

                # Some logging ...
                if random.random()>0.5: clear_output()

                if batch_num is not None:
                    if self.verbose: logger.info('Batch {}, attempt {} -> {}'.format(batch_num, attempt, url))
                else:
                    if self.verbose: logger.info('Attempt {} -> {}'.format(attempt, url))

                
                # -------------------------------------------------------------
                # Submit the request and check for issues.
                
                data = req(url, **api_args).json()

                # Check for any error message in the data returned.
                if 'message' in data: 
                    if self.verbose: logger.info((batch_num, data['message']))      # Log the message.
                    raise Exception(data['message'])  # Force a retry.
                # -------------------------------------------------------------

                
                # Some more logging ...
                if type(data) is list:
                    if batch_num is not None:
                        if self.verbose: logger.info('Batch {}, found {} items'.format(batch_num, len(data)))
                    else:
                        if self.verbose: logger.info('Found {} items'.format(len(data)))
                else:
                    if batch_num is not None:
                        if self.verbose: logger.info('Batch {}, found {}'.format(batch_num, str(data)[:75]))
                    else:
                        if self.verbose: logger.info('Found {}'.format(str(data)[:75]))
                    
    
                return data
            
            except:
                # Catch any exceptionsd and retry after an increasing pause.
                # Note the increasing delays up to max_attempts*2 seconds.
                time.sleep((attempt+1)*2)
                continue

        if self.verbose: logger.info('Failed -> {}'.format(attempt, url))
        return False


    # -------------------------------------------------------------------------
    def item_to_series(self, item):

        # Convert paper obj to dict and add seed indicator.
        item = dict(item)
    
        # Populate a new dict that is suitable for conversion to a series.
        dict_for_df = {}
    
        # For each key, value in r ...
        for key, value in item.items():
    
            # If it;s an atomic value (number or string) then transfer as is.
            if (type(value) is int) | (type(value) is bool) | (type(value) is str):
                dict_for_df[key] = value
    
            # If its a dict then unpack it and transfer its key, values to the 
            # new dict.
            elif type(value) is dict:
                for k, v in value.items(): dict_for_df[k] = v
    
            # If its a list the we need to check whats in the list ...
            elif (type(value) is list):
    
                if len(value):
                    
                    # If its a list of strings then transfer directly.
                    if type(value[0]) is str:
                        dict_for_df[key] = value
        
                    # If its a list of dicts then transfer the values directly; 
                    # check for Nones as we do this.
                    elif type(value[0]) is dict:
                        dict_for_df[key] = [
                            v 
                            for v in [list(d.values())[0] for d in value] 
                            if v is not None
                        ]
    
                else:
                    dict_for_df[key] = []
     
        return pd.Series(dict_for_df)

    def items_to_dataframe(self, items, pool_size=16):

        with Pool(pool_size) as p:
            converted = p.map(self.item_to_series, items)
        
        return pd.DataFrame(converted)





    # -------------------------------------------------------------------------

    def scrape_authors(self, ids, batch_sizes=[100, 10], pool_size=4):
        """Scrape a list of author ids in decreasing batch sizes."""
    
        author_fields = [
            'authorId' ,'externalIds' ,'name' ,'affiliations'
            ,'paperCount' ,'citationCount' ,'hIndex' ,'papers.paperId'
        ]

        authors_df = pd.DataFrame()
        missing_ids = ids
        
        for batch_size in batch_sizes:

            if self.verbose: logger.info('Trying a batch size of {} for {:,} ids'.format(batch_size, len(missing_ids)))
    
            # First, get the basic paper data; no citation/reference lists
            authors = self.get_authors_in_batches(missing_ids, fields=author_fields, batch_size=batch_size, pool_size=pool_size)
            authors_df = pd.concat([authors_df, self.items_to_dataframe(authors)], ignore_index=True)

            # Update the ids to those that are still missing.
            # One issue with this is that SS can return a different id from the one asked for. 
            # This means that we might continue register an id as missing.
            missing_ids = list(set(ids).difference(set(authors_df['authorId'].unique())))

            # No more missing ids so we are finished.
            if (len(authors_df)==len(ids)) | (len(missing_ids)==0): break

        return authors_df.drop_duplicates(subset=['authorId'])




    def scrape_papers_in_chunks(self, ids, batch_sizes=[500, 100], pool_size=10, paper_fields=paper_fields, chunk_size=1_000_000, start_chunk=0, tmp_dir='', tmp_label=''):
        """For large sets of papers; scrape in chunks, save intermediate dfs, recombine."""

        chunks = list(sliced(ids, chunk_size))

        logger.info('Scraping {} chunks of size {}'.format(len(chunks), chunk_size))

        # Scrape the chunks and save as temp files.
        for i, chunk_ids in enumerate(chunks[start_chunk:]):
            filename = tmp_dir+'{}_{}.feather'.format(tmp_label, i+start_chunk)
                
            chunk_df = ss.scrape_papers(chunk_ids, batch_sizes=[500, 100], paper_fields=paper_fields)
            chunk_df.to_feather(filename)

        # recombing the chunks into a single df for return.
        chunk_files = glob(tmp_dir+'{}_*.feather'.format(tmp_label))

        logger.info('Recombining {} chunks'.format(len(chunk_files)))

        chunk_dfs = []

        for chunk_file in chunk_files:
        
            clear_output()
            logger.info(chunk_file)

            # Load the chunk and add it to the list of dfs.
            chunk_dfs.append(pd.read_feather(chunk_file))

            # remove the chunk.
            os.remove(chunk_file)

        # return the concatenated df
        return pd.concat(chunk_dfs, ignore_index=True)           

                    


   
    
    def scrape_papers(self, ids, batch_sizes=[500, 100, 10], pool_size=10, paper_fields=paper_fields):
        """Scrape a list of paper ids in decreasing batch sizes."""
    
        # paper_fields = [
        #     'paperId', 'title', 'url', 'venue', 'publicationVenue', 'year', 'journal', 'isOpenAccess',
        #     'publicationTypes', 'publicationDate',
        #     'fieldsOfStudy',
        #     'abstract',
        #     'authors.authorId', 
        #     'externalIds',
        #     'citationStyles',
        #     'citationCount', 'influentialCitationCount', 'citations.paperId',
        #     'referenceCount', 'references.paperId'
        # ]

        papers_df = pd.DataFrame()
        missing_ids = ids

        # Try increasingly small batches to catch the missing papers, if there are any.
        for batch_size in batch_sizes:

            if self.verbose: logger.info('Trying a batch size of {} for {:,} ids'.format(batch_size, len(missing_ids)))
    
            # First, get the basic paper data; no citation/reference lists
            papers = self.get_papers_in_batches(missing_ids, fields=paper_fields, batch_size=batch_size, pool_size=pool_size)

            if self.verbose: logger.info('converting paper items into dataframe.')
            papers_df = pd.concat([papers_df, self.items_to_dataframe(papers)], ignore_index=True)

            # Update the ids to those that are still missing.
            # One issue with this is that SS can return a different id from the one asked for. 
            # This means that we might continue register an id as missing.
            missing_ids = list(set(ids).difference(set(papers_df['paperId'].unique())))

            # No more missing ids so we are finished.
            if (len(papers_df)==len(ids)) | (len(missing_ids)==0): break

        return papers_df.drop_duplicates(subset=['paperId'])
    


    def is_missing(papers_df, col):
        count_col = '{}Count'.format(col[:-1])
        return papers_df[col].map(len)<papers_df[count_col]
        
    def scrape_missing_citations(self, papers_df, col='citations', is_missing=is_missing, batch_sizes=[500, 100, 10], pool_size=4):

        papers_df = papers_df.set_index('paperId')
        
        fields = ['paperId', '{}.paperId'.format(col)]
        count_col = '{}Count'.format(col[:-1])

        # Current item count
        prev_count = 0
        curr_count = papers_df[col].map(len).sum()
        
        for batch_size in batch_sizes:

            # Find papers with missing citations.
            with_missing_items = is_missing(papers_df, col)

            if self.verbose: logger.info('{} papers with missing {}; prev = {}, curr = {}'.format(with_missing_items.sum(), col, prev_count, curr_count))
            
            # If there are no missing items then we are done or we did not get any new
            # items on the last pass.
            if (with_missing_items.sum()==0) | (curr_count==prev_count): break
                
            # Otherwise try to scrape the missing items
            missing_ids = list(papers_df[with_missing_items].index.unique())
            missing_df = self.items_to_dataframe((
                self.get_papers_in_batches(missing_ids, fields=fields, batch_size=batch_size, pool_size=pool_size)
            )).set_index('paperId')

            # If we found no missing items then we are done..
            if len(missing_df)==0: break

            # Otherwise, add the existing citations/refs to the missing items df.
            missing_df = missing_df.join(papers_df[[col]].add_prefix('current_'), how='left')

            # Combine the new and existing items
            missing_df[col] = missing_df.apply(lambda r: list(set(r[col]).union(r['current_'+col])), axis=1)

            # Replace existing with the combined.
            papers_df.loc[missing_df.index, col] = missing_df[col]

            # Update the count of items.
            prev_count = curr_count
            curr_count = papers_df[col].map(len).sum()

        # Update the counts col
        papers_df[col] = papers_df[col].map(lambda col: col if hasattr(col, '__len__') else [])
        papers_df['scraped_'+count_col] = papers_df[col].map(len)
        
        # Finished. Return the updated df.
        return papers_df.reset_index()
                
                
                                                                             

                
                

        



        
