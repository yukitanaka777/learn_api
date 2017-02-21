# -*- encoding:utf-8 -*-

import sqlite3
import json
import math

#base_img = list
#sqlite3.register_adapter(base_img, lambda l: ';'.join([str(i) for i in l]))
#sqlite3.register_converter("base_img", lambda s: [str(i) for i in s.split(';')])

def db_insert(base_string,name,Id,con):
  
  c = con.cursor()
  result = con.execute("select*from actorsSet")
  label = len(result.fetchall())
  ImgArr = []
  sql = 'insert into actorsSet (id,name,label,data,model_id) values (?,?,?,?,?)'
  for data in base_string:
    try:
      string = base_string[data]
      first_coma = base_string[data].find(',')
      ImgArr.append(string[first_coma:])

    except:
      print "Failed save"

  c.execute(sql,(Id,name,label,ImgArr,label))
  con.commit()
  return label
