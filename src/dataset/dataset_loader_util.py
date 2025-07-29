

class DatasetLoaderUtil:
    
    # torchtext datasets
    @staticmethod
    def text_collate_fn(batch):
        """
        Collate function for text datasets.
        Merges a list of (label, text) tuples into lists.
        """
        labels, texts = zip(*batch)
        return list(labels), list(texts)