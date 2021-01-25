objects = data/raw/training_files.tar \
		  data/raw/secret_data.tar

chan1 = data/interim/chan1/background_chan1_7.79.csv \
		data/interim/chan1/glgl1400_neutralino1100_chan1.csv\
		data/interim/chan1/glgl1600_neutralino800_chan1.csv\
		data/interim/chan1/monojet_Zp2000.0_DM_50.0_chan1.csv\
		data/interim/chan1/monotop_200_A_chan1.csv\
		data/interim/chan1/sqsq_sq1800_neut800_chan1.csv\
		data/interim/chan1/sqsq1_sq1400_neut800_chan1.csv\
		data/interim/chan1/stlp_st1000_chan1.csv\
		data/interim/chan1/stop2b1000_neutralino300_chan1.csv\
		data/interim/chan1/unlabeled_combined_chan1.csv\

chan2a = data/interim/chan2a/background_chan2a_309.6.csv\
         data/interim/chan2a/chaneut_cha200_neut50_chan2a.csv\
		 data/interim/chan2a/chaneut_cha250_neut150_chan2a.csv\
		 data/interim/chan2a/chaneut_cha300_neut100_chan2a.csv\
		 data/interim/chan2a/chaneut_cha400_neut200_chan2a.csv\
		 data/interim/chan2a/gluino_1000.0_neutralino_1.0_chan2a.csv\
		 data/interim/chan2a/pp23mt_50_chan2a.csv\
		 data/interim/chan2a/pp24mt_50_chan2a.csv\
		 data/interim/chan2a/unlabeled_combined_chan2a.csv

chan2b = data/interim/chan2b/background_chan2b_7.8.csv\
		 data/interim/chan2b/chacha_cha300_neut140_chan2b.csv\
		 data/interim/chan2b/chacha_cha400_neut60_chan2b.csv\
		 data/interim/chan2b/chacha_cha600_neut200_chan2b.csv\
		 data/interim/chan2b/chaneut_cha200_neut50_chan2b.csv\
		 data/interim/chan2b/chaneut_cha250_neut150_chan2b.csv\
		 data/interim/chan2b/gluino_1000.0_neutralino_1.0_chan2b.csv\
		 data/interim/chan2b/pp23mt_50_chan2b.csv\
		 data/interim/chan2b/pp24mt_50_chan2b.csv\
		 data/interim/chan2b/stlp_st1000_chan2b.csv\
		 data/interim/chan2b/unlabeled_combined_chan2b.csv

chan3 = data/interim/chan3/background_chan3_8.02.csv\
 		data/interim/chan3/glgl1400_neutralino1100_chan3.csv\
		data/interim/chan3/glgl1600_neutralino800_chan3.csv\
		data/interim/chan3/gluino_1000.0_neutralino_1.0_chan3.csv\
		data/interim/chan3/monojet_Zp2000.0_DM_50.0_chan3.csv\
		data/interim/chan3/monotop_200_A_chan3.csv\
		data/interim/chan3/monoV_Zp2000.0_DM_1.0_chan3.csv\
		data/interim/chan3/sqsq_sq1800_neut800_chan3.csv\
		data/interim/chan3/sqsq1_sq1400_neut800_chan3.csv\
		data/interim/chan3/stlp_st1000_chan3.csv\
		data/interim/chan3/stop2b1000_neutralino300_chan3.csv\
		data/interim/chan3/unlabeled_combined_chan3.csv

chan1npz = $(addprefix data/interim/chan1/, $(addsuffix .npz, $(basename $(notdir $(chan1)))))
chan2anpz = $(addprefix data/interim/chan2a/, $(addsuffix .npz, $(basename $(notdir $(chan2a)))))
chan2bnpz = $(addprefix data/interim/chan2b/, $(addsuffix .npz, $(basename $(notdir $(chan2b)))))
chan3npz = $(addprefix data/interim/chan3/, $(addsuffix .npz, $(basename $(notdir $(chan3)))))
channpz = $(chan1npz) $(chan2anpz) $(chan2bnpz) $(chan3npz)

all: $(objects) $(chan1) $(chan2a) $(chan2b) $(chan3) $(channpz)

$(chan1): data/raw/training_files.tar
	cp data/interim/training_files/chan1/$(notdir $@) $@

$(chan2a): data/raw/training_files.tar
	cp data/interim/training_files/chan2a/$(notdir $@) $@

$(chan2b): data/raw/training_files.tar
	cp data/interim/training_files/chan2b/$(notdir $@) $@

$(chan3): data/raw/training_files.tar
	cp data/interim/training_files/chan3/$(notdir $@) $@

%.tar:
	curl https://zenodo.org/record/3961917/files/training_files.tar?download=1 > $@
	tar -xf $@ -C data/interim/

$(channpz): $(addsuffix .csv,  $(basename $@))
	python src/data/ConvertToNpz.py --input=$(addsuffix .csv,  $(basename $@)) --output=$@
