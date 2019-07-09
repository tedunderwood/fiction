from hathitrust_api import DataAPI
import pandas as pd
import codecs, time, os, random

meta = pd.read_csv('../meta/finalmeta.csv')

data_api = DataAPI("546686df08", "fbda40a272c61f76a40d3c80ee2e")

# already = ['uc2.ark:/13960/fk81j97v57', 'mdp.39015013747905', 'njp.32101067407930']

files = [x for x in os.listdir('../newtexts/') if x.endswith('.txt')]

already = []

for f in files:
	f = f.replace('.txt', '')
	already.append(f)

print(already)
print()

hathi_ids = [x for x in meta.docid if '.' in x]
random.shuffle(hathi_ids)

for anid in hathi_ids:
	if anid in already:
		continue
	pages = []
	finished = False

	anid = anid.replace('+', ':').replace('=', '/')

	print(anid)
	for i in range(1, 1050):
		try:
			text = data_api.getpageocr(anid, i)
			text = codecs.decode(text, 'utf-8')
			pages.append(text)

		except Exception as inst:
			print(type(inst))
			print(inst.args)
			if inst.args[0].startswith('404') or i > 50:
				already.append(anid)
				finished = True
			break
		time.sleep(1.15)

	if finished and i > 20:
		print(i)
		print()
		cleanid = anid.replace(':', '+').replace('/', '=')
		with open('hathitexts/' + cleanid + '.txt', mode = 'w', encoding = 'utf-8') as f:
			for i, p in enumerate(pages):
				f.write(p)
				f.write('\n')
				f.write('<page ' + str(i + 1) + '>\n')
	else:
		time.sleep(10)
		print('failed: ' + str(i))
		print()
		cleanid = anid.replace(':', '+').replace('/', '=')
		if not os.path.isdir(cleanid):
			os.mkdir(cleanid)
		for i, p in enumerate(pages):
			with open(cleanid + '/' + str(i) + '.txt', mode = 'w', encoding = 'utf-8') as f:
				f.write(p)



