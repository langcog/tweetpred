#Python code to download and parse JSON data from a series of MLB games

import urllib
import json

outfolder = '/home/gdoyle/scratch/seetweet/gnip/data/bb/json/'
infile = '/home/gdoyle/scratch/seetweet/gnip/data/bb/json/2014postseason.games.csv'
outcsv = '/home/gdoyle/scratch/seetweet/gnip/data/bb/json/2014postseason.atbats.csv'

urlframe = 'http://gd2.mlb.com/components/game/mlb/year_%s/month_%.2d/day_%.2d/gid_%s_%.2d_%.2d_%smlb_%smlb_1/game_events.json'

rf = open(infile,'r')
header = True
hdict = {}

wf = open(outcsv,'w')
wf.write('gameid,away,home,inning,halfinning,atbatnum,outs_at_end,away_runs_at_end,home_runs_at_end,batter,pitcher,onbase1_at_end,onbase2_at_end,onbase3_at_end,atbat_timez,first_pitch_timecode,last_pitch_timecode,pitches,event_text,event_class\n')
for line in rf:
	if header:
		sline = line.strip().split(',')
		for i in range(0,len(sline)):
			hdict[sline[i]]=i
		header = False
		continue
	sline = line.strip().split(',')
	year = sline[hdict['year']]
	month = int(sline[hdict['month']])
	date = int(sline[hdict['date']])
	away = sline[hdict['away']].lower()
	home = sline[hdict['home']].lower()
	
	print "Downloading", away, "vs.", home, month, date
	
	gameurl = urlframe % (year, month, date, year, month, date, away, home)
	gameid = '%s_%.2d_%.2d_%s_%s' % (year, month, date, away, home)
	outjson = outfolder+gameid+'.json'
	
	#try:
	#	urllib.urlretrieve(gameurl, outjson)
	#except urllib2.HTTPError as e:
	#	print e.read()
	
	print "Loading data from JSON"
	
	jsonf = open(outjson,'r')
	
	for line in jsonf:
		data = line.strip()
		if data:					#remove blank lines
			j = json.loads(data)
	
	jsonf.close()
	
	print "Processing data from JSON"
	
	#Stuff that's consistent throughout game (gameid,away,home)
	gamelinestart = ','.join([gameid,away,home])
	
	lastawayscore = '0'	#away/home runs only reported when they've changed in some files, so store most recent reported score
	lasthomescore = '0'
	
	innings = j['data']['game']['inning']
	for inning in innings:
		inum = inning['num']
		for half in ['top','bottom']:
			if 'atbat' not in inning[half].keys():
				print "No",half,"half of inning",inum
				print "\tscore was",ab.get('away_team_runs',lastawayscore),"<",ab.get('home_team_runs',lasthomescore)
				continue
			atbats = inning[half]['atbat']
			for ab in atbats:
				if isinstance(atbats,dict):
					print "Only one at-bat in this half-inning. Should be bottom of 9th, run scoring. Check below:"
					print '\t',half,inum
					print '\t',atbats['des']
					ab = atbats
				#gameid,home,away,
				if 'pitch' not in ab.keys():
					numpitches = 0
					fptime = ''
					lptime = ''
					print "No pitches in atbat",ab['num'],"-",half,"of inning",inum
					print "\t",ab['des']
				else:
					pitches = ab['pitch']
					if isinstance(pitches,list):
						numpitches = len(pitches)
						fptime = pitches[0]['sv_id']
						lptime = pitches[-1]['sv_id']
					elif isinstance(pitches,dict):
						numpitches = 1
						fptime = pitches['sv_id']
						lptime = pitches['sv_id']
					else:
						raise ValueError("'pitch' variable is neither a list or dict. Lord knows what it is.")
				if len(ab.get('away_team_runs',''))>0:
					lastawayscore = ab['away_team_runs']
				if len(ab.get('home_team_runs',''))>0:
					lasthomescore = ab['home_team_runs']
				outlist = [gamelinestart,inum,half,ab.get('num',''),ab.get('o',''),ab.get('away_team_runs',''),ab.get('home_team_runs',''),ab.get('batter',''),ab.get('pitcher',''),ab.get('b1',''),ab.get('b2',''),ab.get('b3',''),ab.get('start_tfs_zulu',''),fptime,lptime,str(numpitches),'"'+ab.get('des','')+'"',ab.get('event','')]
				wf.write(','.join(outlist)+'\n')
				if isinstance(atbats,dict):
					break			#If only the one at-bat, break out of this inning

rf.close()
wf.close()

