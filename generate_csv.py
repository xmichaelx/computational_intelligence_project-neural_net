import sys
from statistics import median, stdev
comment = "#"
mjd_offset = 2400000


if __name__ == "__main__":
	filename = sys.argv[1]
	columns = sys.argv[2].split(",")
	frame_grades = sys.argv[3].split(",")
	t0 = float(sys.argv[4]) - mjd_offset
	period = float(sys.argv[5])
	output_filename = sys.argv[6]
	lines = None

	with open(filename) as fp:
		lines = list(fp)

	header = [x for x in lines if "#     HJD" in x][0][1:].split()
	items = [dict(zip(header, x.split())) for x in lines if not x.startswith("#")]
	items = [x for x in items if x["GRADE"] in frame_grades]

	items = [[x[key] for key in columns] for x in items]
	dates = [((float(x[0]) - t0) % period) / period for x in items]

	values = [float(x[1]) for x in items]
	items = [[str(x[0])] + list(x[1]) for x in zip(dates, [x[1:] for x in items])]



	with open(output_filename,"w") as fp:
		for item in items:
			print(",".join(item), file=fp)
	