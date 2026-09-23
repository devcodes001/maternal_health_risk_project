from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import joblib, pandas as pd, sys, json
from pathlib import Path
from datetime import datetime, date, timedelta
from database import init_db, get_db

sys.path.insert(0, str(Path(__file__).parent / 'src'))
try:
    from features import add_clinical_features, BASE_COLS, ENGINEERED_COLS
except:
    BASE_COLS = ["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate"]
    ENGINEERED_COLS = []
    def add_clinical_features(df): return df

app = Flask(__name__)
app.secret_key = 'maternal-care-ai-2026-secret'
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / 'models' / 'best_model.joblib'
_bundle = None

def get_model():
    global _bundle
    if _bundle is None:
        try: _bundle = joblib.load(MODEL_PATH)
        except: pass
    return _bundle

def weeks_and_day(lmp_str):
    if not lmp_str:
        return None, None
    try:
        lmp = datetime.strptime(lmp_str,'%Y-%m-%d').date()
        d = (date.today()-lmp).days
        return max(d//7,0), max(d%7,0)
    except: return None, None

def due_date_str(lmp_str):
    if not lmp_str: return None
    try:
        lmp = datetime.strptime(lmp_str,'%Y-%m-%d').date()
        return (lmp+timedelta(days=280)).strftime('%B %d, %Y')
    except: return None

def trimester(w):
    if w is None: return None
    return '1st Trimester' if w<=13 else ('2nd Trimester' if w<=27 else '3rd Trimester')

def enhanced_risk_score(data):
    s = 0.0
    if data.get('systolic',0)>=140: s+=2.5
    elif data.get('systolic',0)>=130: s+=1.0
    if data.get('diastolic',0)>=90: s+=2.2
    elif data.get('diastolic',0)>=80: s+=0.5
    bs = data.get('bs',0)
    if bs>=11: s+=2.5
    elif bs>=7.8: s+=0.7
    age = data.get('age',25)
    if age>=35: s+=1.5
    elif age<18: s+=1.2
    hb = data.get('hemoglobin',12)
    if hb<9: s+=2.0
    elif hb<11: s+=0.8
    if data.get('body_temp',98)>=100: s+=1.2
    if data.get('heart_rate',75)>=100: s+=0.8
    up = data.get('urine_protein',0)
    s += up*0.9
    if data.get('swelling',0): s+=0.8
    if data.get('headache',0): s+=0.9
    if data.get('vision_change',0): s+=1.5
    bmi = data.get('bmi',22)
    if bmi>=30: s+=0.7
    elif bmi<18.5: s+=0.5
    if s>=4.8: return 'high risk', round(s,1)
    if s>=2.2: return 'mid risk', round(s,1)
    return 'low risk', round(s,1)

# ── Helpers ──────────────────────────────────────────
def pid(): return session.get('patient_id')
def require_login(): return redirect(url_for('setup')) if not pid() else None

# ── Routes ───────────────────────────────────────────
@app.route('/')
def dashboard():
    if not pid(): return redirect(url_for('setup'))
    db = get_db(); p = pid(); today_s = date.today().isoformat()
    latest = db.execute('SELECT * FROM assessments WHERE patient_id=? ORDER BY assessed_at DESC LIMIT 1',(p,)).fetchone()
    upcoming = db.execute('SELECT * FROM appointments WHERE patient_id=? AND date>=? AND is_completed=0 ORDER BY date,time LIMIT 4',(p,today_s)).fetchall()
    recent_entries = db.execute('SELECT * FROM journal_entries WHERE patient_id=? ORDER BY entry_date DESC LIMIT 3',(p,)).fetchall()
    stats = {
        'assessments': db.execute('SELECT COUNT(*) FROM assessments WHERE patient_id=?',(p,)).fetchone()[0],
        'appointments': db.execute('SELECT COUNT(*) FROM appointments WHERE patient_id=?',(p,)).fetchone()[0],
        'journal': db.execute('SELECT COUNT(*) FROM journal_entries WHERE patient_id=?',(p,)).fetchone()[0],
        'clinic_visits': db.execute('SELECT COUNT(*) FROM clinic_visits WHERE patient_id=?',(p,)).fetchone()[0],
    }
    water_today = db.execute('SELECT * FROM water_intake WHERE patient_id=? AND date=?',(p,today_s)).fetchone()
    meds_active = db.execute('SELECT * FROM medications WHERE patient_id=? AND is_active=1',(p,)).fetchall()
    recent_diet = db.execute('SELECT * FROM diet_logs WHERE patient_id=? AND date=? ORDER BY logged_at DESC',(p,today_s)).fetchall()
    recent_activity = db.execute('SELECT * FROM activity_logs WHERE patient_id=? AND date=? ORDER BY logged_at DESC',(p,today_s)).fetchall()
    db.close()
    lmp = session.get('lmp_date'); wk,dy = weeks_and_day(lmp)
    return render_template('dashboard.html',
        latest_assessment=latest, upcoming=upcoming, recent_entries=recent_entries,
        stats=stats, water_today=water_today, meds_active=meds_active,
        recent_diet=recent_diet, recent_activity=recent_activity,
        pregnancy_week=wk, pregnancy_day=dy, due_date=due_date_str(lmp), trimester=trimester(wk))

@app.route('/setup', methods=['GET','POST'])
def setup():
    if request.method=='POST':
        name=request.form.get('name','').strip()
        if not name: flash('Please enter your name.','danger'); return redirect(url_for('setup'))
        db=get_db()
        cur=db.execute('INSERT INTO patients (name,age,lmp_date,blood_group,phone,doctor_name,hospital,height_cm,weight_kg) VALUES (?,?,?,?,?,?,?,?,?)',
            (name,request.form.get('age'),request.form.get('lmp_date'),request.form.get('blood_group'),
             request.form.get('phone'),request.form.get('doctor_name'),request.form.get('hospital'),
             request.form.get('height_cm') or None, request.form.get('weight_kg') or None))
        db.commit(); pid_new=cur.lastrowid; db.close()
        session.update({'patient_id':pid_new,'patient_name':name,'lmp_date':request.form.get('lmp_date','')})
        flash(f'Welcome, {name}!','success'); return redirect(url_for('dashboard'))
    return render_template('setup.html')

@app.route('/assessment', methods=['GET','POST'])
def assessment():
    r=require_login();
    if r: return r
    result=None
    if request.method=='POST':
        try:
            data={k:float(request.form[k]) for k in ['age','systolic','diastolic','bs','bodytemp','heartrate']}
            data['weight_kg']=float(request.form.get('weight_kg') or 0) or None
            data['hemoglobin']=float(request.form.get('hemoglobin') or 12)
            data['urine_protein']=int(request.form.get('urine_protein',0))
            data['swelling']=1 if request.form.get('swelling') else 0
            data['headache']=1 if request.form.get('headache') else 0
            data['vision_change']=1 if request.form.get('vision_change') else 0
            h=data.get('weight_kg') or 60; ht=float(request.form.get('height_cm') or 158)
            data['bmi']=round(h/(ht/100)**2,1) if h and ht else 22
            label,score=enhanced_risk_score({'systolic':data['systolic'],'diastolic':data['diastolic'],
                'bs':data['bs'],'age':data['age'],'hemoglobin':data['hemoglobin'],
                'body_temp':data['bodytemp'],'heart_rate':data['heartrate'],
                'urine_protein':data['urine_protein'],'swelling':data['swelling'],
                'headache':data['headache'],'vision_change':data['vision_change'],'bmi':data['bmi']})
            db=get_db()
            db.execute('INSERT INTO assessments (patient_id,age,systolic_bp,diastolic_bp,bs,body_temp,heart_rate,weight_kg,hemoglobin,urine_protein,swelling,headache,vision_change,risk_level,risk_score,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (pid(),data['age'],data['systolic'],data['diastolic'],data['bs'],data['bodytemp'],data['heartrate'],
                 data['weight_kg'],data['hemoglobin'],data['urine_protein'],data['swelling'],data['headache'],data['vision_change'],label,score,request.form.get('notes','')))
            db.commit(); db.close()
            result={'risk_level':label,'score':score,'data':data}
        except Exception as e: flash(f'Error: {e}','danger')
    db=get_db()
    history=db.execute('SELECT * FROM assessments WHERE patient_id=? ORDER BY assessed_at DESC LIMIT 10',(pid(),)).fetchall()
    db.close()
    return render_template('assessment.html',result=result,history=history)

@app.route('/calendar')
def calendar():
    r=require_login();
    if r: return r
    db=get_db()
    apps=db.execute('SELECT * FROM appointments WHERE patient_id=? ORDER BY date,time',(pid(),)).fetchall()
    db.close(); apps_list=[dict(a) for a in apps]
    return render_template('calendar.html',appointments=apps_list,appointments_json=json.dumps(apps_list))

@app.route('/api/appointments',methods=['GET','POST','DELETE','PATCH'])
def api_appointments():
    if not pid(): return jsonify({'error':'Not logged in'}),401
    db=get_db()
    if request.method=='GET':
        a=db.execute('SELECT * FROM appointments WHERE patient_id=? ORDER BY date,time',(pid(),)).fetchall()
        db.close(); return jsonify([dict(x) for x in a])
    if request.method=='POST':
        d=request.json
        db.execute('INSERT INTO appointments (patient_id,title,date,time,type,doctor,notes) VALUES (?,?,?,?,?,?,?)',
            (pid(),d['title'],d['date'],d.get('time',''),d.get('type','routine'),d.get('doctor',''),d.get('notes','')))
        db.commit(); db.close(); return jsonify({'success':True})
    if request.method=='DELETE':
        db.execute('DELETE FROM appointments WHERE id=? AND patient_id=?',(request.args.get('id'),pid()))
        db.commit(); db.close(); return jsonify({'success':True})
    if request.method=='PATCH':
        db.execute('UPDATE appointments SET is_completed=1 WHERE id=? AND patient_id=?',(request.args.get('id'),pid()))
        db.commit(); db.close(); return jsonify({'success':True})

@app.route('/journal')
def journal():
    r=require_login();
    if r: return r
    db=get_db()
    entries=db.execute('SELECT * FROM journal_entries WHERE patient_id=? ORDER BY entry_date DESC',(pid(),)).fetchall()
    db.close(); lmp=session.get('lmp_date'); wk,dy=weeks_and_day(lmp)
    return render_template('journal.html',entries=entries,pregnancy_week=wk,pregnancy_day=dy)

@app.route('/api/journal',methods=['GET','POST','DELETE'])
def api_journal():
    if not pid(): return jsonify({'error':'Not logged in'}),401
    db=get_db()
    if request.method=='GET':
        e=db.execute('SELECT * FROM journal_entries WHERE patient_id=? ORDER BY entry_date DESC',(pid(),)).fetchall()
        db.close(); return jsonify([dict(x) for x in e])
    if request.method=='POST':
        d=request.json; lmp=session.get('lmp_date'); wk,dy=weeks_and_day(lmp)
        ed=d.get('entry_date',date.today().isoformat())
        if lmp:
            try:
                delta=(datetime.strptime(ed,'%Y-%m-%d').date()-datetime.strptime(lmp,'%Y-%m-%d').date()).days
                wk=max(delta//7,0); dy=max(delta%7,0)
            except: pass
        db.execute('INSERT INTO journal_entries (patient_id,pregnancy_week,pregnancy_day,title,content,mood,symptoms,baby_movements,entry_date) VALUES (?,?,?,?,?,?,?,?,?)',
            (pid(),wk,dy,d.get('title',''),d['content'],d.get('mood','neutral'),json.dumps(d.get('symptoms',[])),d.get('baby_movements',0),ed))
        db.commit(); db.close(); return jsonify({'success':True})
    if request.method=='DELETE':
        db.execute('DELETE FROM journal_entries WHERE id=? AND patient_id=?',(request.args.get('id'),pid()))
        db.commit(); db.close(); return jsonify({'success':True})

# ── Wellness ──────────────────────────────────────────
@app.route('/wellness')
def wellness():
    r=require_login();
    if r: return r
    db=get_db(); today_s=date.today().isoformat()
    water=db.execute('SELECT * FROM water_intake WHERE patient_id=? AND date=?',(pid(),today_s)).fetchone()
    diet_today=db.execute('SELECT * FROM diet_logs WHERE patient_id=? AND date=? ORDER BY logged_at',(pid(),today_s)).fetchall()
    activity_today=db.execute('SELECT * FROM activity_logs WHERE patient_id=? AND date=? ORDER BY logged_at',(pid(),today_s)).fetchall()
    diet_week=db.execute('SELECT date,COUNT(*) as meals FROM diet_logs WHERE patient_id=? AND date>=? GROUP BY date ORDER BY date',
        (pid(),(date.today()-timedelta(days=6)).isoformat())).fetchall()
    activity_week=db.execute('SELECT date,SUM(duration_minutes) as mins FROM activity_logs WHERE patient_id=? AND date>=? GROUP BY date ORDER BY date',
        (pid(),(date.today()-timedelta(days=6)).isoformat())).fetchall()
    water_week=db.execute('SELECT date,cups_consumed FROM water_intake WHERE patient_id=? AND date>=? ORDER BY date',
        (pid(),(date.today()-timedelta(days=6)).isoformat())).fetchall()
    total_activity=sum(a['duration_minutes'] or 0 for a in activity_today)
    db.close()
    return render_template('wellness.html',
        water=water,diet_today=diet_today,activity_today=activity_today,
        total_activity=total_activity,
        diet_week=json.dumps([dict(x) for x in diet_week]),
        activity_week=json.dumps([dict(x) for x in activity_week]),
        water_week=json.dumps([dict(x) for x in water_week]))

@app.route('/api/water',methods=['POST'])
def api_water():
    if not pid(): return jsonify({'error':'Not logged in'}),401
    d=request.json; today_s=date.today().isoformat(); db=get_db()
    cups=d.get('cups',1); action=d.get('action','add')
    existing=db.execute('SELECT * FROM water_intake WHERE patient_id=? AND date=?',(pid(),today_s)).fetchone()
    if existing:
        new_cups=max(0,(existing['cups_consumed']+(cups if action=='add' else -cups)))
        db.execute('UPDATE water_intake SET cups_consumed=? WHERE patient_id=? AND date=?',(new_cups,pid(),today_s))
    else:
        db.execute('INSERT INTO water_intake (patient_id,date,cups_consumed,goal_cups) VALUES (?,?,?,8)',(pid(),today_s,max(0,cups)))
    db.commit(); db.close(); return jsonify({'success':True})

@app.route('/api/diet',methods=['POST','DELETE'])
def api_diet():
    if not pid(): return jsonify({'error':'Not logged in'}),401
    db=get_db()
    if request.method=='POST':
        d=request.json
        db.execute('INSERT INTO diet_logs (patient_id,date,meal_type,food_items,portion,notes) VALUES (?,?,?,?,?,?)',
            (pid(),d.get('date',date.today().isoformat()),d.get('meal_type','snack'),d['food_items'],d.get('portion','medium'),d.get('notes','')))
        db.commit(); db.close(); return jsonify({'success':True})
    if request.method=='DELETE':
        db.execute('DELETE FROM diet_logs WHERE id=? AND patient_id=?',(request.args.get('id'),pid()))
        db.commit(); db.close(); return jsonify({'success':True})

@app.route('/api/activity',methods=['POST','DELETE'])
def api_activity():
    if not pid(): return jsonify({'error':'Not logged in'}),401
    db=get_db()
    if request.method=='POST':
        d=request.json
        db.execute('INSERT INTO activity_logs (patient_id,date,activity_type,duration_minutes,intensity,steps,notes) VALUES (?,?,?,?,?,?,?)',
            (pid(),d.get('date',date.today().isoformat()),d.get('activity_type','walking'),d.get('duration_minutes',30),d.get('intensity','light'),d.get('steps',0),d.get('notes','')))
        db.commit(); db.close(); return jsonify({'success':True})
    if request.method=='DELETE':
        db.execute('DELETE FROM activity_logs WHERE id=? AND patient_id=?',(request.args.get('id'),pid()))
        db.commit(); db.close(); return jsonify({'success':True})

# ── Medications ───────────────────────────────────────
@app.route('/medications')
def medications():
    r=require_login();
    if r: return r
    db=get_db()
    meds=db.execute('SELECT * FROM medications WHERE patient_id=? ORDER BY is_active DESC,created_at DESC',(pid(),)).fetchall()
    today_s=date.today().isoformat()
    logs=db.execute('SELECT ml.*,m.name,m.color FROM medication_logs ml JOIN medications m ON ml.medication_id=m.id WHERE ml.patient_id=? AND ml.date=?',(pid(),today_s)).fetchall()
    db.close()
    return render_template('medications.html',medications=meds,today_logs=logs,medications_json=json.dumps([dict(m) for m in meds]))

@app.route('/api/medications',methods=['GET','POST','DELETE','PATCH'])
def api_medications():
    if not pid(): return jsonify({'error':'Not logged in'}),401
    db=get_db()
    if request.method=='GET':
        m=db.execute('SELECT * FROM medications WHERE patient_id=? AND is_active=1',(pid(),)).fetchall()
        db.close(); return jsonify([dict(x) for x in m])
    if request.method=='POST':
        d=request.json
        db.execute('INSERT INTO medications (patient_id,name,dosage,frequency,times,start_date,end_date,color,purpose,notes) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (pid(),d['name'],d.get('dosage',''),d.get('frequency','once_daily'),json.dumps(d.get('times',['08:00'])),
             d.get('start_date',date.today().isoformat()),d.get('end_date',''),d.get('color','#667eea'),d.get('purpose',''),d.get('notes','')))
        db.commit(); db.close(); return jsonify({'success':True})
    if request.method=='PATCH':
        mid=request.args.get('id'); d=request.json or {}
        if d.get('toggle'): db.execute('UPDATE medications SET is_active=CASE WHEN is_active=1 THEN 0 ELSE 1 END WHERE id=? AND patient_id=?',(mid,pid()))
        db.commit(); db.close(); return jsonify({'success':True})
    if request.method=='DELETE':
        db.execute('DELETE FROM medications WHERE id=? AND patient_id=?',(request.args.get('id'),pid()))
        db.commit(); db.close(); return jsonify({'success':True})

@app.route('/api/med-log',methods=['POST'])
def api_med_log():
    if not pid(): return jsonify({'error':'Not logged in'}),401
    d=request.json; db=get_db(); today_s=date.today().isoformat()
    existing=db.execute('SELECT * FROM medication_logs WHERE medication_id=? AND patient_id=? AND date=? AND scheduled_time=?',
        (d['medication_id'],pid(),today_s,d.get('scheduled_time',''))).fetchone()
    if existing:
        db.execute('UPDATE medication_logs SET taken=1,taken_at=? WHERE id=?',(datetime.now().isoformat(),existing['id']))
    else:
        db.execute('INSERT INTO medication_logs (medication_id,patient_id,date,scheduled_time,taken,taken_at) VALUES (?,?,?,?,1,?)',
            (d['medication_id'],pid(),today_s,d.get('scheduled_time',''),datetime.now().isoformat()))
    db.commit(); db.close(); return jsonify({'success':True})

# ── Clinic Visits ────────────────────────────────────
@app.route('/clinic-visits',methods=['GET','POST'])
def clinic_visits():
    r=require_login();
    if r: return r
    if request.method=='POST':
        f=request.form; db=get_db()
        db.execute('INSERT INTO clinic_visits (patient_id,date,doctor_name,hospital,pregnancy_week,weight_kg,bp_systolic,bp_diastolic,hemoglobin,urine_protein,fetal_heart_rate,fundal_height,baby_position,doctor_notes,patient_concerns,prescriptions,next_visit_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (pid(),f.get('date'),f.get('doctor_name'),f.get('hospital'),f.get('pregnancy_week') or None,
             f.get('weight_kg') or None,f.get('bp_systolic') or None,f.get('bp_diastolic') or None,
             f.get('hemoglobin') or None,f.get('urine_protein','negative'),
             f.get('fetal_heart_rate') or None,f.get('fundal_height') or None,
             f.get('baby_position'),f.get('doctor_notes'),f.get('patient_concerns'),
             f.get('prescriptions'),f.get('next_visit_date')))
        db.commit(); db.close(); flash('Clinic visit saved!','success'); return redirect(url_for('clinic_visits'))
    db=get_db()
    visits=db.execute('SELECT * FROM clinic_visits WHERE patient_id=? ORDER BY date DESC',(pid(),)).fetchall()
    db.close(); lmp=session.get('lmp_date'); wk,_=weeks_and_day(lmp)
    return render_template('clinic_visits.html',visits=visits,pregnancy_week=wk)

# ── Guide ─────────────────────────────────────────────
@app.route('/guide')
def guide():
    r=require_login();
    if r: return r
    db=get_db()
    latest=db.execute('SELECT risk_level FROM assessments WHERE patient_id=? ORDER BY assessed_at DESC LIMIT 1',(pid(),)).fetchone()
    db.close(); current_risk=latest['risk_level'] if latest else None
    return render_template('guide.html',current_risk=current_risk)

# ── Report ────────────────────────────────────────────
@app.route('/report')
def report():
    r=require_login();
    if r: return r
    db=get_db(); p=pid()
    patient=db.execute('SELECT * FROM patients WHERE id=?',(p,)).fetchone()
    assessments=db.execute('SELECT * FROM assessments WHERE patient_id=? ORDER BY assessed_at',(p,)).fetchall()
    appointments=db.execute('SELECT * FROM appointments WHERE patient_id=? ORDER BY date',(p,)).fetchall()
    entries=db.execute('SELECT * FROM journal_entries WHERE patient_id=? ORDER BY entry_date',(p,)).fetchall()
    visits=db.execute('SELECT * FROM clinic_visits WHERE patient_id=? ORDER BY date',(p,)).fetchall()
    meds=db.execute('SELECT * FROM medications WHERE patient_id=? ORDER BY created_at',(p,)).fetchall()
    water_hist=db.execute('SELECT * FROM water_intake WHERE patient_id=? ORDER BY date',(p,)).fetchall()
    diet_hist=db.execute('SELECT date,COUNT(*) as meals FROM diet_logs WHERE patient_id=? GROUP BY date ORDER BY date',(p,)).fetchall()
    activity_hist=db.execute('SELECT date,SUM(duration_minutes) as mins FROM activity_logs WHERE patient_id=? GROUP BY date ORDER BY date',(p,)).fetchall()
    db.close()
    all_syms=[]
    for e in entries:
        try: all_syms.extend(json.loads(e['symptoms'] or '[]'))
        except: pass
    sym_counts={}
    for s in all_syms: sym_counts[s]=sym_counts.get(s,0)+1
    sym_counts=dict(sorted(sym_counts.items(),key=lambda x:x[1],reverse=True))
    risk_counts={'low risk':0,'mid risk':0,'high risk':0}
    for a in assessments: risk_counts[a['risk_level']]=risk_counts.get(a['risk_level'],0)+1
    completed=sum(1 for a in appointments if a['is_completed'])
    lmp=session.get('lmp_date'); wk,_=weeks_and_day(lmp)
    return render_template('report.html',
        patient=patient,assessments=assessments,appointments=appointments,
        entries=entries,visits=visits,meds=meds,water_hist=water_hist,
        diet_hist=diet_hist,activity_hist=activity_hist,
        symptom_counts=sym_counts,risk_counts=risk_counts,
        risk_progression=json.dumps([{'date':a['assessed_at'][:10],'risk':a['risk_level']} for a in assessments]),
        completed_appointments=completed,total_appointments=len(appointments),
        pregnancy_week=wk,due_date=due_date_str(lmp),trimester=trimester(wk))

if __name__=='__main__':
    init_db()
    print("[OK] MaternalCare AI running at http://127.0.0.1:5000")
    app.run(debug=True,port=5000)
