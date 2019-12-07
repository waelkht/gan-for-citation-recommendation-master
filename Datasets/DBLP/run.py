from __builtin__ import True

class Paper:
    def __init__(self, id, title, authors, venue, year, abstract, citations):
        self.id = id
        self.title = title
        self.authors = authors 
        self.venue = venue 
        self.year = year 
        self.abstract = abstract
        self.citations = citations
        
    def str(self):
        return "{}, {}, {}, {}, {}, {}".format(self.id, self.title, self.authors, self.venue, self.year, self.citations)

"""
Generates a stream of Papers
##### INPUTS ####
:dir_name     The name of the DBNL-Structure
"""
def paper_generator(dir_name):
    f = open(dir_name, 'r')
    
    pap_id = None
    authors = []
    venue = None
    year = None
    abstract = ""
    title = None
    citations = []
    
    for l in f:
        if l == "\n": # new paper
            yield Paper(pap_id, title, authors, venue, year, abstract, citations)
            pap_id = None
            authors = []
            venue = None
            year = None
            abstract = ""
            title = None
            citations = []
        elif l[0] == '#': #its some information
            if l[1] == '*': # TITLE
                title = l[2::].strip()
            elif l[1] == 't': # Year
                year = l[2::].strip()
            elif l[1] == 'i': # id
                pap_id = l[6::].strip()
            elif l[1] == '!': # Abstract
                abstract = l[2::].strip()
            elif l[1] == 'c': # Venue
                venue = l[2::].strip()
            elif l[1] == '%': # Citation
                citations.append(l[2::].strip())
            elif l[1] == '@': # Authors
                authors_raw = l[2::].strip()
                if len(authors_raw) > 0:
                    authors_raw = authors_raw.split(",")
                    authors.extend([a.strip().replace(" ", "") for a in authors_raw])
            else:
                print "Can't recognize: ", l
        else:
            print "Can't recognize: ", l
    
"""
This function creates the Abstract Files, acl.txt, acl-metadata.txt, author-citation-network.txt, author_ids.txt
"""
def create_aan_structure(dir_name):
    gen = paper_generator(dir_name)
    
    INFORMATION_RETRIEVAL = ["SIGIR", "ACL", "EACL", "ECIR", "NAACL", "CIKM", "EMNLP", "COLING"] # 7_946
    MACHINE_LEARNING = ["NIPS", "ICML", "SIGKDD", "WSDM", "ICDE", "PAKDD"] # 3_546
    COMPUTER_VISION = ["CVPR", "ECCV", "ICCV", "ACCV", "ICIP", "ICPR", "MM"] # 6_732
    NET_AND_COM = ["COM", "SIGCOMM", "ICC", "GLOBECOM", "MOBICOM", "ICDCS", "SECON", "ICNP"] # 5_924
    SECURITY = ["SP", "NDSS", "FC", "ACSAC", "ARES", "ISI"] # 1_180
    # 7_946 + 3_546 + 6_732 + 5_924 + 1_180 = 25_328
    CONFERENCES = INFORMATION_RETRIEVAL + MACHINE_LEARNING + COMPUTER_VISION + NET_AND_COM + SECURITY
    #CONFERENCES = INFORMATION_RETRIEVAL
    
    # helping variables
    authors = dict() # stores an individual ID for every author
    acl = open("raw/acl.txt", 'w')
    pan = open("raw/paper_author_network.txt", 'w')
    acl_meta = open("raw/acl-metadata.txt", 'w')
    author_ids = open("raw/author_ids.txt", "w")
    
    rel_paper_count = 0
    total_paper_count = 0
    
    for paper in gen:
        total_paper_count += 1
        if paper.title == None or paper.venue == None or paper.id == None:
            continue
        else:
            # check if the paper is from one of the important ones.
            is_relevant = False
            for conf in CONFERENCES:
                if conf.lower() in paper.venue.replace(".", "").replace("\'", "").lower().split():
                    is_relevant = True
                    break 
            if not is_relevant:
                continue
            
            # if the paper is relevant:
            rel_paper_count += 1
            
            # add the authors
            for a in paper.authors:
                if not authors.has_key(a):
                    authors[a] = len(authors)
                    author_ids.write("{}\t{}\n".format(len(authors), a))
                # ... and write it in the paper-author-network
                pan.write("{} ==> {}\n".format(paper.id, authors[a]))
                
            # print the citations
            for c in paper.citations:
                acl.write("{} ==> {}\n".format(paper.id, c))
                
            # at last, we print in the acl-metadata file
            author_str = ""
            for a in paper.authors:
                author_str = author_str + "; {}".format(a)
            author_str = author_str[2::]
            
            acl_meta.write("id = {{{}}}\n".format(paper.id))
            acl_meta.write("author = {{{}}}\n".format(author_str))
            acl_meta.write("title = {{{}}}\n".format(paper.title))
            acl_meta.write("venue = {{{}}}\n".format(paper.venue))
            acl_meta.write("year = {{{}}}\n".format(paper.year))
            acl_meta.write("\n".format(paper.year))
            
            # write the abstract in the maching file
            f = open("raw/papers_text/{}.txt".format(paper.id), "w")
            f.write(paper.abstract)
            f.close()
    
    acl.close()
    pan.close()
    acl_meta.close()
    
    print "processed {} papers. {} are relevant.".format(total_paper_count, rel_paper_count)

if __name__ == "__main__":
    create_aan_structure("DBLPOnlyCitationOct19.txt")
    print "[INFO] Finished!"