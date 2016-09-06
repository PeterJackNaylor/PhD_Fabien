import os


def DeconvNet(path, split, data_gene, classifier_name, nber_output):
    if split != " val":
	    ScriptContent = open(path).read()
	    ScriptContent = ScriptContent.replace('SSPLIT', split)
	    ScriptContent = ScriptContent.replace('CLASSIFIERNAME', classifier_name)
	    ScriptContent = ScriptContent.replace('datagen', data_gene)
	    ScriptContent = ScriptContent.replace('NBER_OUTPUT', nber_output)
	else: 
		ScriptContent = open(path).read()
		ScriptContent = ScriptContent.replace('NBER_OUTPUT', nber_output)
	return ScriptContent

def make_net(wd, data_gene_train, data_gene_test, 
	         path_train, path_val,
	         classifier_name="DeconvNet", nber_output="2"):
    with open(os.path.join(wd, 'train.prototxt'), 'w') as f:
        f.write(str(DeconvNet('train', data_gene_train, classifier_name, nber_output)))
    with open(os.path.join(wd, 'test.prototxt'), 'w') as f:
        f.write(str(DeconvNet('test', data_gene_test, classifier_name, nber_output)))
    with open(os.path.join(wd, 'deploy.prototxt'), 'w') as f:
        f.write(str(DeconvNet('val', data_gene_train, classifier_name, nber_output)))
