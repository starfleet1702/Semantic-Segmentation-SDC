import os;
import time;

cont_file_path = './cont.txt';
log_file_path = './logs/';

def can_continue():
    can_cont = True;
    if os.path.exists(cont_file_path):
        with open(cont_file_path,'r') as f:
            line = f.readline();
            if(line=='Y'):
                print("Cont...");
                can_cont = True;
            elif(line=='N'):
                print("| Stop |");
                can_cont = False;
    return can_cont;
          
# can_continue();

def create_log():
    if(not os.path.exists(log_file_path)):
        os.mkdir(log_file_path);
    log_file = os.path.join(log_file_path , str(str(time.time())+'.txt'));
    file = open(log_file,"w");
    file.close();
    return log_file;

def log_n_print(log_file,line):
    with open(log_file,'a') as f:
        f.writelines(line+'\n');
        print(line);
    
# log_f = create_log();
# print(log_f)
# log_line(log_f,"1....a");
# log_line(log_f,"2....a");
# log_line(log_f,"3....a");


