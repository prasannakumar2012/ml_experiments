import boto3
import os
bucket_name = "prascf-prascf-t9dth25rsrg"
string = "from ec2"
encoded_string = string.encode("utf-8")
file_name = "hello_ec2.txt"
s3.Bucket(bucket_name).put_object(Key=file_name, Body=encoded_string)
os.system("shutdown now -h")
