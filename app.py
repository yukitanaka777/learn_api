# -*- encoding:utf-8 -*-

from flask import Flask,render_template,redirect,request,url_for,make_response,session,jsonify
from flask.ext.cors import CORS,cross_origin
import sqlite3
import json
import os
import Cookie
import modules.db_controll as dc
import modules.learn as learn
import modules.learn_result as learn_result
import modules.post_img as post

app = Flask(__name__)
CORS(app,resources={r"/save_and_learn":{"origins":"*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
base_img = list
sqlite3.register_adapter(base_img, lambda l: ';'.join([str(i) for i in l]))
sqlite3.register_converter("base_img", lambda s: [str(i) for i in s.split(';')])
con = sqlite3.connect("face_memory.db",detect_types = sqlite3.PARSE_DECLTYPES,check_same_thread=False)
con.row_factory = sqlite3.Row


@app.route('/')
def index():

  print os.environ.get('HTTP_COOKIE',False)
  print "request cookie :"+str(request.cookies.get('Solaris_model_id',False))
  model_id = session.get('Solaris_model_id',False)

  print model_id
  return render_template('index.html')

@app.route('/save_and_learn',methods=['POST','GET','OPTIONS'])
@cross_origin('*')
def save_and_learn():
  if request.method == 'GET':
    
    return redirect('http://192.168.1.6:5000/takePicture')

  elif request.method == 'POST':
    post.moveRequest()
    json_data = request.get_json()
    model_id = dc.db_insert(json_data['img'],json_data['name'],json_data['id'],con)
    result = learn.start(con,model_id)
    post.main(con)
    print "learn model id : "+str(int(model_id))
    try:
      model_id = str(int(model_id))
      return model_id
    except:
      return "no return"

@app.route('/face_result',methods=['GET','POST'])
@cross_origin('*')
def face_result():

  if request.method == 'POST':

    model_id = str(request.form.get('Solaris_model_id'))
    generate_id = learn_result.start(request.form.get('login_base64_file'),model_id,con)
    return redirect('http://192.168.1.6:3333/views?id='+str(generate_id)+'')

  else:

    return redirect('http://192.168.1.6:3333/views')


@app.route('/getUserList',methods=['GET','POST'])
@cross_origin('*')
def getUser():

  if request.method == 'POST':

    return redirect('http://192.168.1.6:8000/')

  else:

    c = con.cursor()
    c.execute('select*from actorsSet')
    result = c.fetchall()
    params_array = {}
    for i,row in enumerate(result):
      param = {}
      param['name'] = row['name']
      param['img'] = row['data'][0]
      params_array[str(i)] = param
      for cer in params_array:
        print params_array[cer]['name']

    return jsonify(data=params_array)


if __name__ == '__main__':
  app.debug = True
  app.secret_key = "hgj[pqkgq4:g[q{:g/[q.gq/4["
  #app.run(host='0.0.0.0',port=9000,threaded=True)
  app.run(host='0.0.0.0',port=9000)


