
#https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
import requests, os
if not os.path.exists("app/javaArchives"): os.mkdir("app/javaArchives")
if not os.path.exists("app/jdkArchives"): os.mkdir("app/jdkArchives")
from urllib.parse import urlparse, unquote
import hashlib

repos =[("junit-team", "junit5")]
headers = {"Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}
GITHUB_API = "https://api.github.com"
token = "" #insert github token here (no checkbox permissions are needed, just for 5000 rate limit instead of 60 for anonymous per hour)
headers["Authorization"] = "Bearer " + token

jdk_url = "https://github.com/openjdk/jdk/archive/refs/heads/master.zip"
jdk_response = requests.get(jdk_url)
if jdk_response.status_code == 200:
    from zipfile import ZipFile
    #from io import BytesIO
    #ZipFile(BytesIO(jdk_response.content)).extractall(path="app/jdkArchives")
    localpath = "app/jdkArchives/" + os.path.basename(urlparse(unquote(jdk_url)).path)
    with open(localpath, "wb") as f:
        f.write(jdk_response.content)
    with open(localpath, "rb") as f:
        ZipFile(f).extractall(path="app/jdkArchives")
    os.remove(localpath)
else: print("HTTP Response error: " + str(jdk_response.status_code))

for owner, repo in repos:
    api_url = GITHUB_API + "/repos/" + owner + "/" + repo + "/pulls"
    page = 1
    while True:
        response = requests.get(api_url, headers=headers,
                                params={"state": "closed", "per_page": "100", "page": str(page)})   
        if response.status_code == 200:
            json = response.json()
            for i, item in enumerate(json):
                print(str((page-1)*100+i+1) + ". " + item["title"])
                for label in item["labels"]:
                    print("Name: " + label["name"] + ("" if label["description"] is None else " Description: " + label["description"]))
                pull_number = item["number"]
                commit_url = GITHUB_API + "/repos/" + owner + "/" + repo + "/pulls/" + str(pull_number) + "/commits"
                commitResponse = requests.get(commit_url, headers=headers, params={"per_page": 1})
                if commitResponse.status_code == 200:
                    commitjson = commitResponse.json()
                    commit_details_url = GITHUB_API + "/repos/" + owner + "/" + repo + "/commits/" + commitjson[0]["sha"]
                    print(commit_details_url)
                    commit_details_response = requests.get(commit_details_url, headers=headers)
                    if commit_details_response.status_code == 200:
                        commitjson = commit_details_response.json()
                        print(commitjson)
                        for file in commitjson["files"]:
                            if not file["raw_url"].endswith(".java"): continue
                            #filename = os.path.basename(urlparse(unquote(file["raw_url"])).path)
                            #print(file["raw_url"], filename)
                            file_response = requests.get(file["raw_url"])
                            if file_response.status_code == 200:
                                m = hashlib.sha1()
                                m.update(file_response.content)
                                grade = 5 if True else 1
                                filename = m.hexdigest() + "_" + str(grade) + ".java"
                                with open("app/javaArchives/" + filename, "wb") as f:
                                    f.write(file_response.content)
                            else: print("HTTP Response error: " + str(file_response.status_code))
                    else: print("HTTP Response error: " + str(commit_details_response.status_code))
                    #assert False
                else: print("HTTP Response error: " + str(commitResponse.status_code))
            if len(json) != 100: break
        else: print("HTTP Response error: " + str(response.status_code)); break
        page += 1