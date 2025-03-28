import os
from dotenv import load_dotenv
import pandas as pd
from minio import Minio
import minio_helper


def list_minio_files(client, bucket_name, fpath):
    
    from minio.error import S3Error
    names=[]
    try:
        objects = client.list_objects(bucket_name, prefix=fpath, recursive=True)
    
        # Print out the names of all the files in the prefix
        for obj in objects:
            names.append(obj.object_name)
        return names
    except S3Error as e:
        print(f"Error occurred: {e}")

def query_omit_errors(df: pd.DataFrame) -> pd.DataFrame:
    return df.query("Error.isnull()")

class GetMinioArticles():

    def __init__(self, subreddit_name='news', input_dir=os.path.join("urls_and_articles","news"), output_dir=''):
        load_dotenv()
        print("CONNECTING TO MINIO 🦩")
        self.client = Minio(
            os.getenv("minio_endpoint"),
            access_key = os.getenv("minio_access_key"),
            secret_key = os.getenv("minio_secret_key"),
            secure = False
        )
    
        self.bucket_name = os.getenv("bucket_name")
    
        self.subreddit_name = subreddit_name

        self.input_dir = input_dir
        self.output_dir = output_dir
        
    def get_articles_from_minio(self, output_filename="scraped_news.csv",my_query=query_omit_errors, concat_df=True, to_minio=False, no_save=False):
        
        output_path = os.path.join(self.output_dir,output_filename)
        
        def _save_files(df, output_path=output_path, to_minio=to_minio):
            if to_minio:
                minio_helper.save_file_minio(self.client, self.bucket_name, output_path, obj_to_upload=df, obj_type='csv')
            else:
                df.to_csv(output_path, index=False)
                print(f"SAVED FILE TO {output_path} 🎁")

        if no_save:
            def _save_files(*args, **kwargs):
                print('NOT SAVING ANYTHING')
                return None
            
        self.file_list = list_minio_files(self.client, self.bucket_name, self.input_dir)

        if concat_df:
            df_list=[]
        print("DOWNLOADING FILES ⬇")
        for file in self.file_list:
            print(f'{file=}')
            
            df = minio_helper.read_file_minio(client = self.client,
                                                bucket_name = self.bucket_name,
                                                obj_name = file,
                                                obj_type = 'csv')
            df=my_query(df)#("Error.isnull()")

            
            if concat_df and len(df)>0:
                df_list.append(df)
            elif not concat_df:
                _save_files(df, output_path=os.path.join(df, sef.output_dir, file.split('/')[-1]))
                
                    
                
        if concat_df:
            df = pd.concat(df_list)
            _save_files(df)
            return df
        return None

def main():
    
    load_dotenv()
    
    print([os.getenv('bucket_name'),os.getenv('minio_endpoint')])
    
    import pandas as pd
    print('TESTING DOWNLOADER')
    def my_query(df: pd.DataFrame) -> pd.DataFrame:
        df = df.query("Error.isnull() and not ArticleText.isnull()").drop_duplicates(subset=['ArticleText'], keep=False)
        
        return df[['Domain','expanded_url', 'created_utc', 'Rating', 'ArticleTitle','ArticleText', 'Keywords']]
        
    df = GetMinioArticles().get_articles_from_minio(my_query=my_query, no_save=False)
    print(df.columns)
    print(f'{len(df)=}')
    print('TEST FINISHED')

if __name__=='__main__':
    main()
