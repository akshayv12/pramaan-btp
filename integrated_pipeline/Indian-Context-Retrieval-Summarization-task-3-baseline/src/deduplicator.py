from datasketch import MinHash, MinHashLSH

class Deduplicator:
    def __init__(self, threshold=0.85, num_perm=128):
        # threshold=0.85 means documents must be 85% similar to be considered duplicates
        # num_perm=128 is the number of mathematical hash permutations (standard for accuracy)
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.database = {}
    
    def _get_ngrams(self, text, n=3):
        """Breaks text into overlapping chunks (shingles) of 3 words."""
        words = text.replace('\n', ' ').split()
        if len(words) < n:
            return [" ".join(words)] if words else []
        return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

    def _generate_signature(self, text):
        """Generates a compressed MinHash signature from the text n-grams."""
        m = MinHash(num_perm=128)
        for ngram in self._get_ngrams(text):
            m.update(ngram.encode('utf8'))
        return m

    def process_document(self, doc_id, text):
        """
        Checks if a document is a duplicate. 
        If it is NEW, it saves it. If it is a DUPLICATE, it drops it.
        """
        signature = self._generate_signature(text)
        
        # Query the LSH index to see if any existing signatures are an 85% match
        duplicate_matches = self.lsh.query(signature)
        
        if duplicate_matches:
            print(f"[DROP] Document '{doc_id}' is a near-duplicate of {duplicate_matches}!")
            return False # Reject payload
        else:
            print(f"[SAVE] Document '{doc_id}' is unique. Adding to database.")
            self.lsh.insert(doc_id, signature)
            self.database[doc_id] = text
            return True # Accept payload

# ==========================================
# TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    dedup = Deduplicator(threshold=0.60)

    # Document 1: A standard scraped article
    doc1 = "The Indian context aware retrieval system processes Hindi and Sanskrit documents efficiently."
    
    # Document 2: The exact same article, but with a few words changed/added (A Near-Duplicate!)
    doc2 = "The Indian context aware retrieval system processes Hindi and Sanskrit documents very efficiently today."
    
    # Document 3: A completely different article
    doc3 = "OpenCV is used for median blur and grayscale transformations in computer vision."

    print("--- Running MinHash LSH Pipeline ---\n")
    dedup.process_document("Doc_1_Original", doc1)
    dedup.process_document("Doc_2_Slightly_Changed", doc2)
    dedup.process_document("Doc_3_Different_Topic", doc3)
