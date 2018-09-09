import json
import boto3

region = 'us-east-1'
ami = 'ami-0ff8a91507f77f867'
# AMI = "ami-24bd1b4b"
instance_type = 't2.micro'
ec2_client = boto3.client('ec2', region_name=region)
ec2 = boto3.resource('ec2')


def lambda_handler(event, context):
    out = event["Records"][0]["Sns"]["Message"]
    init_script = """#!/bin/bash
    aws s3 cp s3://prascf-prascf-t9dth25rsrg/sample_py.py sample_py.py
    sudo python sample_py.py"""
    print (init_script)
    instance = ec2_client.run_instances(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=1, # required by boto, even though it's kinda obvious.
        MaxCount=1,
        InstanceInitiatedShutdownBehavior='terminate', # make shutdown in script terminate ec2
        UserData=init_script # file to run on instance init.
    )

    print ("New instance created.")
    instance_id = instance['Instances'][0]['InstanceId']
    print (instance_id)

    return {
        "out" : event["Records"][0]["Sns"]["Message"],
        "eve" : str(event["Records"]),
        "cont" : str(context),
        "statusCode": 200,
        "body": json.dumps('Hello from Lambda!')
    }
    # TODO implement
