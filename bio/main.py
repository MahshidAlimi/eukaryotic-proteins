from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# print("cyto");
raw_data_cyto = SeqIO.parse("./datasets/cyto.fasta", "fasta");
training_cyto = [(seq_record, "cyto") for seq_record in raw_data_cyto]

# print("\n mito");
raw_data_mito = SeqIO.parse("./datasets/mito.fasta", "fasta");
training_mito = [(seq_record, "mito") for seq_record in raw_data_mito]

# print("\n nucleus");
raw_data_nucleus = SeqIO.parse("./datasets/nucleus.fasta", "fasta");
training_nucleus = [(seq_record, "nucleus") for seq_record in raw_data_nucleus]

# print("\n secreted");
raw_data_secreted = SeqIO.parse("./datasets/secreted.fasta", "fasta");
training_secreted = [(seq_record, "secreted") for seq_record in raw_data_secreted]

# test data
blind_tests = SeqIO.parse("./datasets/blind.fasta", "fasta");

training = training_mito;
training.extend(training_cyto);
training.extend(training_nucleus);
training.extend(training_secreted);

def bio_feat(record):
    clean_seq = str(MutableSeq(record.seq)).replace("X", "")
    clean_seq = clean_seq.replace("U", "C")
    clean_seq = clean_seq.replace("B", "N")
    clean_seq = MutableSeq(clean_seq).toseq()

    ### features
    seq_length = len(str(clean_seq))
    analysed_seq = ProteinAnalysis(str(clean_seq))
    molecular_weight = analysed_seq.molecular_weight()
    amino_percent = analysed_seq.get_amino_acids_percent().values()
    isoelectric_points = analysed_seq.isoelectric_point()
    count = analysed_seq.count_amino_acids().values()
    # aromaticity = analysed_seq.aromaticity()
    instability_index = analysed_seq.instability_index()
    # hydrophobicity = analysed_seq.protein_scale(ProtParamData.kd, 5, 0.4)
    secondary_structure_fraction = analysed_seq.secondary_structure_fraction()

    return np.array([seq_length, molecular_weight, isoelectric_points, instability_index] + list(secondary_structure_fraction) + list(count) + list(amino_percent))

features = ([bio_feat(record) for record, _ in training])
labels = ([label for _, label in training])

### cross validation
## normalize features before SVM
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

### SVM
clf = SVC(probability=True).fit(X_train, y_train)
print("score svm: ", clf.score(X_test, y_test))

## prepare test data
testing1 = [(test.id, bio_feat(test)) for test in blind_tests]
testing = [[id, scaler.transform(test)] for id, test in testing1]
test_features = [f for _,f in testing]
test_ids = [id for id,_ in testing]

## make predictions for blind data set
blind_predictions = clf.predict_proba(test_features)

## print out results
l = ['Cyto', 'Mito', 'Nucleus', 'Secret']
for index in range(20):
    p = blind_predictions[index].tolist()
    print(test_ids[index], l[p.index(max(p))], "Confidence", int(round(max(p) * 100 )), "%")

## try out random forest
### Random Forest
##
# clf_rforest = RandomForestClassifier(n_estimators=10)
# clf_rforest = clf_rforest.fit(X_train, y_train)
# print("score random forest: ", clf_rforest.score(X_test, y_test))
