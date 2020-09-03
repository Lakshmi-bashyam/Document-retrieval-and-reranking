from helper import Dataloader

if __name__=="__main__":
    dataloader = Dataloader()
    query = 'Who is the author of the book, "The Iron Lady: A Biography of Margaret Thatcher"?'
    dataloader.extract_feature(query) 

