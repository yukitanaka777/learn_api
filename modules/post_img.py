# -*- encofing -*-

import urllib
import urllib2
import requests
import sqlite3

url = "http://127.0.0.1:3333/receive_obj"
urls = {"obj":"http://172.20.10.2:3333/receive_obj","move":"http://127.0.0.1:3333/receive_move"}
#base_img = list
#sqlite3.register_converter("base_img", lambda s: [str(i) for i in s.split(';')])
img = None
name = None

def main(con):
  #con = sqlite3.connect('face_memory.db',detect_types = sqlite3.PARSE_DECLTYPES)
  #con.row_factory = sqlite3.Row
  c = con.cursor()
  c.execute('select*from actorsSet')
  rows = c.fetchall()
  for row in rows:
    name = row['name']
    for Img in row['data']:
      img = Img
  
  params = {"img":img,"name":name}
  #params = {"name":name}
  res = requests.post(url,data=params)
  return res


def moveRequest():
  res = requests.post("http://127.0.0.1:3333/receive_move",data="learn_started")
  return res
