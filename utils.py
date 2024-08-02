url='https://drive.google.com/file/d/1r04pmN6u0g379ikpD0cNqlu4HOfPMU7F/view?usp=sharing'

import requests
import zipfile
import os
import io

def download_and_unzip(url, extract_to):
    # ZIP 파일을 다운로드할 임시 파일 경로 설정

    local_zip_path = os.path.join(extract_to, 'melspecs_5sec.zip')

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # ZIP 파일 열기 및 압축 해제
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted files to {extract_to}")

    # 임시 ZIP 파일 삭제
    os.remove(local_zip_path)
    print(f"Deleted temporary ZIP file {local_zip_path}")



zip_file_url = url  # ZIP 파일 다운로드 링크
output_folder = './data/'  # 파일을 추출할 폴더 경로

download_and_unzip(zip_file_url, output_folder)