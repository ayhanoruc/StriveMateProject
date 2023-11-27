import yaml
import os 



def load_yaml(file_path:str, context:str):

    with open(file_path, "r") as f:

        data = yaml.safe_load(f)
        #print(data)
        #print(type(data))
        #print(type(data[context])) 
        return data[context]

def current_time(formatted = False):
    import datetime
    default = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    if formatted:
        formatted = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M%p")
        return formatted
    #parsed_datetime = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M") # convert string to datetime
    return default



if __name__ == "__main__":
    path= os.path.join(os.getcwd(), "instructions", "spr.yaml")

    context= load_yaml(path, "spr_generator")
    print((context))
    print(type(current_time(formatted=False)))

