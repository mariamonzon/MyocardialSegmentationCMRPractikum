import numpy as np
import joblib
import base64
import glob
from pathlib import Path
import torch
import shutil
import imageio
import pandas as pd
import os

def save_data(path, data):
    joblib.dump(data, path)

def load_data(path):
    return joblib.load(path)  # pickle.load(open(path,'rb'))

def make_directory(path, dir_name=""):
    dir_path = Path(path).joinpath(dir_name)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def remove_files(directory='../weights/*'):
    import os, glob
    files = glob.glob(directory)
    for f in files:
        print(f)
        os.remove(f)
    print("Files removed")

def image_to_data_url(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    with open(filename, 'rb') as f:
        img = f.read()
    return prefix + base64.b64encode(img).decode('utf-8')


def save_checkpoint(state, is_best, resume_file, tosave):
    folder_path = make_directory(Path(resume_file).parent)
    torch.save(state, resume_file)
    if is_best:
        shutil.copyfile(resume_file, tosave)  # Save the model if the precision is best

def png_to_gif(path, save_filename, fps=4):
    "Creates  a GIF with all the png files contained in a  directory : path"
    images_gif = []
    for filename in glob.glob("*.png"):
        images_gif.append(imageio.imread(filename))
    imageio.mimsave(path + '/' + save_filename + '.gif', images_gif, fps=fps)


def html(dir_path, filename_pattern, ext='.png', title='dataset_vis'):
    htmlfile = open(Path(dir_path).joinpath(title + '.html') , "w")
    htmlfile.write("<html>\n")
    htmlfile.write("<head>\n")
    htmlfile.write("\t<title>" + str(title) + "</title>\n")
    htmlfile.write("\t<script>\n")
    htmlfile.write("\t\t <function app(val) {document.getElementById('review').innerHTML += val;}\n")
    htmlfile.write("\t</script>\n")
    htmlfile.write("</head>\n")
    htmlfile.write("<body>\n")
    for image_path in glob.iglob(dir_path + '/*/*' + filename_pattern + '*' + ext, recursive=True):
        htmlfile.write('<img src = " ' + image_to_data_url(image_path) + '" alt ="cfg">>\n')
        # htmlfile.write("</p>\n")
    htmlfile.write("</body>\n")
    htmlfile.write("</html>\n")
    htmlfile.close()

def csv_reader(file_name):
    for row in open(file_name, "r"):
        yield row

def generate_excel_from_csv(csv_file, excel_path, sheet_="Sheet1"):

    writer = pd.ExcelWriter(excel_path)
    data = pd.read_csv(csv_file, header=0, index_col=False)
    data.to_excel(writer, sheet_name=sheet_, index=False)
    writer.save()
    writer.close()
