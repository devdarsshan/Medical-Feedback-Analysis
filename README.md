
# Logging in the server
Login to the server with the following snippet
```sh
$ ssh repugen_vm_azure@52.165.187.62
```
# Cloning the directory
Create your own directory in the server and clone the git repository
```sh
$ mkdir <name of the directory>
$ cd <name of the directory>
$ git clone https://github.com/jaiobs/Repugen.git
```

Note: Incase you are cloning the repository in the server. The packages are already installed
in the server. No need to install it again. Use the requirements.txt as a reference to know what packages are used.
# Deploying the code in the server
  - To deploy the code in web server use start_app.sh.
  - Before deploying don't forget to kill the current pipeline with the kill_app.sh. 
    To Kill the current running web server, move to repugen_vm_azure@RepugenVM:~/Christopher/Repugen and then run the following snippets.
    ```sh
    $ cd Christopher/Repugen/
    $ chmod u+r+x  kill_app.sh
    $ ./kill_app.sh
    ```
 - To deploy the code in the web server, move back to your own directory that you have created.
     ```sh
    $ chmod u+r+x  start_app.sh
    $ ./start_app.sh
    ```
Use postman to hit the API's

# Train python file
Note: To train a  larger dataset follow the steps.
- run the train.py file in the console
```sh
    $ python3 train.py "<path of the dataset>"
```
- under <path of the dataset> provide the path where the dataset is present.
- The trained model would be present under upload folder.You could view the trained model 
  using this snippet
```sh
    $ cd upload    
    $ ls
```
