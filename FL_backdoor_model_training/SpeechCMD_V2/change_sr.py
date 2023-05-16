import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
all_dic_name = os.listdir(dir_path)
print(all_dic_name)
for d in all_dic_name:
    if os.path.isdir(os.path.join(dir_path, d)):
        
        all_wav = [_ for _ in os.listdir(os.path.join(dir_path, d)) if _.endswith(".wav")]
        # print(all_wav[0])
        dir = os.path.join(dir_path, d)
        for f in all_wav:
            
            os.system("sox "+ os.path.join(dir, f) + " -r 44100 " + os.path.join(dir, "tmp.wav"))
            os.remove(os.path.join(dir, f))
            os.rename(os.path.join(dir, "tmp.wav"), os.path.join(dir, f))

        print(d + " done.")

