import os
from moviepy.editor import VideoFileClip
import numpy as np
import time
import multiprocessing as mp



class KjnMultiProcessMuvieCuter(object):
    def __init__(self, gap=0.01, cores=8):
        super().__init__()
        self.gap = gap
        self.cores = cores
        self.output_folder = ''

    def _load_movies(self, path):
        images = []
        valid_movies = [".mp4", ".avi"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_movies:
                continue
            images.append(os.path.join(path, f))
        return images

    def _cut_signle_movie(self, movie_path):
        _, tail = os.path.split(movie_path)
        clip = VideoFileClip(movie_path)
        czasmax = clip.duration
        czas = 0
        nklatek = int(czasmax/self.gap)
        outputcore = self.output_folder+tail[7:-4]
        for i in range(0,nklatek):
            output = os.path.join(self.output_folder, tail[7:-4]+"_"+str(i)+".jpg")
            clip.save_frame(output,t=czas)
            czas += self.gap

    def cut_movies_form_folder(self, input_folder, output_folder):
        self.output_folder = output_folder
        movies = self._load_movies(input_folder)
        with mp.Pool(processes=self.cores) as pool:
            pool.map(self._cut_signle_movie, movies)

if (__name__ == '__main__'):
    cores = int(mp.cpu_count()-2)
    print("mp.cpu_count(): ", mp.cpu_count())
    print("cores: ", cores)
    movie_cuter = KjnMultiProcessMuvieCuter(gap=0.04, cores=cores)
    # movie_cuter.cut_movies_form_folder(input_folder='E:/kjn_biedronka/dataset_movies2/',
    #                                      output_folder='test')
    movie_cuter.output_folder = '20200728_191602_cut'
    movie_cuter._cut_signle_movie('E:/kjn_biedronka/dataset_movies1/20200728_191602.mp4')
