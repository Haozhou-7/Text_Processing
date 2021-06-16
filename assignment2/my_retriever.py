import math
class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    # Generate two-dimension dictionary of all terms in the form like:
    # {Doc1: {Term1: number of occurrences, Term2: number of occurrences}}
    # We could use this dictionary to calculate size of document vector.
    def compute_document_vector_size(self):
        self.rev_dictss = {}
        for (k, v) in self.index.items():
            for vk, vv in v.items():
                self.addTwoDimict(self.rev_dictss, vk, k, vv)
        return self.rev_dictss
    # First calculate the document frequency of all terms,
    # then calculate their inverse document frequency(idf), save the results in a one dimension dictionary.
    def compute_iverse_document_frequency(self):
        self.df = {}
        self.idf = {}
        # Calculate the document frequency
        for (k, v) in self.index.items():
            self.df[k] = len(self.index[k])
        # Calculate idf
        for idx in self.df.keys():
            self.idf[idx] = math.log(self.num_docs / self.df[idx], 10)
        return self.idf
    # Function to update content in two-dimensional dictionary.
    def addTwoDimict(self, thedict, key_a, key_b, val):
        if key_a in thedict:
            thedict[key_a].update({key_b: val})
        else:
            thedict.update({key_a:{key_b: val}})
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        d = {}
        cosqd = {}
        dsize = self.compute_document_vector_size()
        self.query = query
        # Unlike "compute_document_vector_size" function, the following double loop generates 
        # two-dimension dictionary of "queries' terms" in the form like:
        # {Doc1: {Term1: number of occurrences, Term2: number of occurrences}}
        # We could use this dictionary to calculate numerator part of the (cosine)vector-space model.
        for letter in self.query:
            for doc in self.doc_ids:
                # Queries' letters must be in original index.
                if self.index.keys().__contains__(letter):
                    if self.index[letter].__contains__(doc):
                        self.addTwoDimict(d, doc, letter, self.index[letter][doc])
        # When we assign binary weights values for terms in document (and query) vectors,
        # the numerator part of the (cosine)vector-space model would only be the number of 
        # terms of query in each document.
        # Meanwhile, the denomiantor part ot this model would only be the size of document vector.
        # The size of the query vector is constant across comparisons, so can be dropped without
        # affecting how candidates are ranked(Same in tf and tfidf).
        if self.term_weighting == "binary":
            # Get cosine value of the document.
            for (key, value) in d.items():
                cosqd[key] = len(d[key]) / math.sqrt(len(dsize[key]))
        # When we assign term frequency values for terms in document (and query) vectors,
        # We need to calculate the numerator part by the number of times each term appears in the query and the document, 
        # and the denominator part is still calculated by the document size.
        if self.term_weighting == "tf":
            for (key, value) in d.items():
                sum_qd = 0
                num = 0
                # The collection of the number of occurrences of all terms in each document.
                d_val = list(d[key].values())
                # The collection of all terms in the document.
                d_second_key = list(d[key].keys())
                for idx in range(len(d_val)):
                    # Calculate number of times each term appears in the query.
                    for query_letter in self.query:
                        if d_second_key[idx] == query_letter:
                            num = num +1
                    # Calculate the numerator part.
                    sum_qd = sum_qd + (num * d_val[idx])                
                # Calculate the denominator part.
                sum_val = 0
                dsize_val = list(dsize[key].values())
                for idx in range(len(dsize_val)):
                    sum_val = sum_val + dsize_val[idx] ** 2
                d_abs = math.sqrt(sum_val)
                # Get cosine value of the document.
                cosqd[key] = sum_qd / d_abs
        # When we assign tfidf values for terms in document (and query) vectors,
        # We only need to multiply each item in the numerator and denominator of fraction of tf by the idf value.     
        if self.term_weighting == "tfidf":        
            # Get idf value for each term.
            idf_val = self.compute_iverse_document_frequency()
            for (key, value) in d.items():
                sum_qd = 0
                num = 0
                d_val = list(d[key].values())
                d_second_key = list(d[key].keys())
                for idx in range(len(d_val)):
                    # Calculate number of times each term appears in the query.
                    for query_letter in self.query:
                        if d_second_key[idx] == query_letter:
                            num = num +1
                    # Calculate the numerator part.
                    sum_qd = sum_qd + (idf_val[d_second_key[idx]] * num *
                                       d_val[idx] * idf_val[d_second_key[idx]])
                # Calculate the denominator part.
                sum_val = 0
                dsize_val = list(dsize[key].values())
                dsize_val_second_key = list(dsize[key].keys())
                for idx1 in range(len(dsize_val)):
                    # Multiply the term whose weight in document is not 0 (it appears at least once) by idf
                    d_tfidf = (dsize_val[idx1] * idf_val[dsize_val_second_key[idx1]]) ** 2
                    sum_val = sum_val + d_tfidf
                d_tfidf_abs = math.sqrt(sum_val)
                # Get cosine value of the document.
                cosqd[key] = sum_qd / d_tfidf_abs
        # Sort the resulting dictionary in descending order of cosine value.        
        cosqd_order = sorted(cosqd.items(), key = lambda x : x[1], reverse=True)   
        # Returns the first 10 document numbers with the largest cosine value.
        # That is, the top ten documents most relevant to the query.
        return list((cosqd_order[0][0], cosqd_order[1][0], cosqd_order[2][0], cosqd_order[3][0], 
                     cosqd_order[4][0], cosqd_order[5][0], cosqd_order[6][0], cosqd_order[7][0], 
                     cosqd_order[8][0], cosqd_order[9][0]))


