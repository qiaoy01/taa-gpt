# TAA GPT
TAAGPT is a small and easy-to-understand GPT kernel that includes a trainer, a web interface for loss functions, and a small training model. By installing TAAGPT locally, you can learn the basic structure and training methods of the Transformer. <br>

To install TAAGPT, follow these steps:<br>

From python official side: https://www.python.org/ to Install Python 3.10.11. Please be noted to use version 3.10 not 3.11.<br>

Use pip install -r requirements.txt to install Torch, CUDA, and other required packages.<br>

Run python main.py to start the web service<br>

Start your browser and navigate to http://127.0.0.1:8000 to run and view the training process.<br>

Please note that this project is still in its initial stages and additional features will be added gradually. The GPT kernel will also be modified accordingly. If you find any errors, please feel free to point them out and make changes. <br>

This project is released under the MIT License. <br>

## Version 0.2.0 Release notes <br>
- Users can now initiate training directly from the web page without the need to run the training program separately. <br>
- Users can obtain configuration files on the web page. 
- Users can select training programs on the web page as well. 
- The web page allows editing of model configuration files and displays the total parameter size of the model. 
- In addition to the adder program trainer as an example, the built-in openwebtext training set includes 8 million English articles. 
- The interface and font have been updated to be even more fancy. 
- Over 40 replaceable background images are provided, including themes such as mathematics, space, future, and robots for users to freely use. These background images were created by the author and have no copyright requirements, so they can be used freely. Of course, users cannot claim their copyright.
