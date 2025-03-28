import pickle as pickle
import pandas as pd
import sys
import os
import io
import zipfile
import tempfile

def create_credentials(access_key, secret_key, user_name=None):
    if user_name is None:
        user_name = access_key
    user_info = {'access_key': access_key,
                 'secret_key': secret_key,
                 'user_name': user_name}
    with open(os.getcwd() + '/minio.pkl', 'wb') as handle:
        pickle.dump(user_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

def init_minio(access_key, secret_key, boto):
    if boto:
        import boto3
        print('hi')
        client = boto3.client(
                                service_name ='s3',
                                endpoint_url="http://10.51.2.249:9001",
                                aws_access_key_id=access_key,
                                aws_secret_access_key=secret_key,
                                region_name='us-east-1'
                             )

    else:
        from minio import Minio
        from minio.error import S3Error
        client = Minio(
            os.getenv("minio_endpoint"),
            access_key = os.getenv("minio_access_key"),
            secret_key = os.getenv("minio_secret_key"),
            secure = False
        )
    return client

def list_files(fpath):
    from minio.error import S3Error
    root_idx = find_nth(os.getcwd(), '/', 3)
    if root_idx > 0:
        root_path = os.getcwd()[:root_idx]  + '/README'
    else:
        root_path = os.getcwd() + '/README'
    print(root_path)
    
    with open(root_path + '/minio.pkl', 'rb') as handle:
        user_info = pickle.load(handle)

    client = init_minio(user_info['access_key'], user_info['secret_key'], False)    

    bucket_name = user_info['user_name']+"-processed"
    names = []
    try:
        objects = client.list_objects(bucket_name, prefix=fpath, recursive=True)
    
        # Print out the names of all the files in the prefix
        for obj in objects:
            names.append(obj.object_name)
        return names
    except S3Error as e:
        print(f"Error occurred: {e}")



    
def read_file(fpath, obj_type=None, personal=True):
    if not personal:
        return read_lab_data(fpath, obj_type)
        
    boto = use_boto()
    root_idx = find_nth(os.getcwd(), '/', 3)
    if root_idx > 0:
        root_path = os.getcwd()[:root_idx]  + '/README'
    else:
        root_path = os.getcwd() + '/README'
    print(root_path)
    
    with open(root_path + '/minio.pkl', 'rb') as handle:
        user_info = pickle.load(handle)

    client = init_minio(user_info['access_key'], user_info['secret_key'], boto)    

    bucket_name = user_info['user_name']+"-processed"
    
    if boto:
        return read_file_boto(client, bucket_name, fpath ,obj_type)
    else:
        return read_file_minio(client, bucket_name, fpath ,obj_type)

def read_lab_data(fpath, obj_type=None):
    boto = use_boto()
    root_idx = find_nth(os.getcwd(), '/', 3)
    if root_idx > 0:
        root_path = os.getcwd()[:root_idx]  + '/README'
    else:
        root_path = os.getcwd() + '/README'
    print(root_path)
    
    with open(root_path + '/minio.pkl', 'rb') as handle:
        user_info = pickle.load(handle)

    client = init_minio(user_info['access_key'], user_info['secret_key'], boto)    

    bucket_name = fpath[:fpath.find('/')]
    fname = fpath[fpath.find('/')+1:]
    
    if boto:
        return read_file_boto(client, bucket_name, fname ,obj_type)
    else:
        return read_file_minio(client, bucket_name, fname ,obj_type)

def read_file_boto(client, bucket_name, obj_name ,obj_type=None):
    try:
        response = client.get_object(Bucket=bucket_name, Key=obj_name)
        data = response['Body'].read()

        if obj_type == 'pickle':
            return pickle.loads(data)
        elif obj_type == 'csv':
            data_io = io.BytesIO(data)
            return pd.read_csv(data_io)
        else:
            return io.BytesIO(data)
            
    except Exception as e:
        print(f"Error occurred: {e}")


def read_file_minio(client, bucket_name, obj_name ,obj_type=None):
    from minio.error import S3Error
    try:

        if obj_type=='shp':
            
            response = client.get_object(bucket_name, obj_name[:obj_name.rfind('/')])
            shapefile_bytes = io.BytesIO(response.read())
            
            # Step 4: Extract the ZIP file into a temporary directory
            with tempfile.TemporaryDirectory() as tmpdirname:
                with zipfile.ZipFile(shapefile_bytes, 'r') as z:
                    z.extractall(tmpdirname)
                
                # Step 5: Locate the .shp file inside the temporary directory
                for filename in os.listdir(tmpdirname):
                    if (filename.endswith(".shp")) and (filename == obj_name[obj_name.rfind('/')+1:]):
                        print(filename)
                        shapefile_path = os.path.join(tmpdirname, filename)
                
                # Step 6: Load the shapefile into a GeoDataFrame using geopandas
                import geopandas as gpd
                gdf = gpd.read_file(shapefile_path)
                return gdf
                
        response = client.get_object(bucket_name, obj_name)
        
        # Read data from the object
        data = response.read()
        response.close()
        response.release_conn()
        
        # Deserialize the pickle data
        if obj_type == 'pickle':
            return pickle.loads(data)
        elif obj_type == 'csv':
            data_io = io.BytesIO(data)
            return pd.read_csv(data_io)
        elif obj_type == 'zst':
            data_io = io.BytesIO(data)
            return pd.read_json(data_io, compression = dict(method='zstd',
                                                            max_window_size = 2147483648),
                                                            lines = True,
                                                            nrows = 100)
        else:
            return io.BytesIO(data)
            
    except S3Error as e:
        print(f"Error occurred: {e}")
        return

def save_file(minio_fpath, obj_to_upload, obj_type=None):
    boto = use_boto()
    root_idx = find_nth(os.getcwd(), '/', 3)
    if root_idx > 0:
        root_path = os.getcwd()[:root_idx]  + '/README'
    else:
        root_path = os.getcwd() + '/README'
    print(root_path)
    with open(root_path + '/minio.pkl', 'rb') as handle:
        user_info = pickle.load(handle)

    client = init_minio(user_info['access_key'], user_info['secret_key'], boto)    

    bucket_name = user_info['user_name']+"-processed"
    
    
    if boto:    
        return save_file_boto(client, bucket_name, minio_fpath, obj_to_upload,obj_type)
    else:
        return save_file_minio(client, bucket_name, minio_fpath, obj_to_upload, obj_type)

def save_file_minio(client, bucket_name, minio_fpath, obj_to_upload, obj_type):
    
    if not client.bucket_exists(bucket_name):
        print(f"Bucket '{bucket_name}' does not exist.")
        return

    if obj_type == 'csv':
        csv_data = obj_to_upload.to_csv(index=False)
        data_stream = io.BytesIO(csv_data.encode('utf-8'))
    elif obj_type=='pbf':
        try:
            minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=obj_to_upload,
                file_path=minio_fpath,
                content_type="application/x-protobuf"  # MIME type for .pbf files
            )
            print(f"'{object_name}' successfully uploaded to '{bucket_name}'.")
        except S3Error as e:
            print(f"Error occurred: {e}")
    else:
        if obj_type != 'BytesIO':
            pickle_data = pickle.dumps(obj_to_upload)
            data_stream = io.BytesIO(pickle_data)
    
    # Get the size of the data
    data_stream_size = data_stream.getbuffer().nbytes

    try:
        # Upload the variable data
        client.put_object(
            bucket_name=bucket_name,
            object_name=minio_fpath,
            data=data_stream,
            length=data_stream_size
        )
        print(f"Variable data is successfully uploaded as '{minio_fpath}' to bucket '{bucket_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_file_boto(client, bucket_name, minio_fpath, obj_to_upload, obj_type):
    if obj_type == 'csv':
        csv_data = obj_to_upload.to_csv(index=False)
        data_stream = BytesIO(csv_data.encode('utf-8'))
    elif obj_type == 'pickle':
        pickle_data = pickle.dumps(obj_to_upload)
        data_stream = io.BytesIO(pickle_data)
    elif obj_type == 'image':
        client.put_object(
            Bucket=bucket_name,        # The bucket name in MinIO
            Key=minio_fpath,            # Object name (file name in MinIO)
            Body=obj_to_upload,                  # BytesIO object containing the plot
            ContentType='image/png'     # MIME type
        )
        print(f"Variable data is successfully uploaded as '{minio_fpath}' to bucket '{bucket_name}'.")
        return

    
    try:
        # Upload the variable data
        client.put_object(Bucket=bucket_name, Key=minio_fpath, Body=data_stream)
        print(f"Variable data is successfully uploaded as '{minio_fpath}' to bucket '{bucket_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

##### helper #######

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def use_boto():
    major = sys.version_info.major
    minor = sys.version_info.minor
    
    # Check if the version is 3.7 or higher
    if (major == 3 and minor >= 7) or (major > 3):
        # >= 3.7 can use minio
        return False
    else:
        # < 3.7, doesn't have __future__ annotations; use boto3
        return True
    
