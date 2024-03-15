const os = require("os");
const fs = require("fs");
const path = require("path");
// read a json file from the bash argument and print it
const file = process.argv[2];
const data = fs.readFileSync(file, "utf8");
const json = JSON.parse(data);
const out = json.log_history
	.filter((log) => "eval_BLEU_bleu" in log)
	.map((log) => {
		log.eval_BLEU_precisions = undefined;
		return log;
	});

const selectedFields = ["epoch", "eval_loss", "eval_BLEU_bleuP", "eval_ROUGE_rouge1", "eval_ROUGE_rouge2", "eval_ROUGE_rougeL"];
const delim = ",";
const csvLines = out.map(log => Object.entries(log).filter(([key]) => selectedFields.includes(key)).map(([_, v]) => v).join(delim));

const csvHeader = Object.keys(out[0]).filter(k => selectedFields.includes(k));
csvLines.unshift(csvHeader.join(delim));
const csvOut = csvLines.join(os.EOL);
console.log(csvOut)

// write the json to a new file next to where it was read
const newFile = path.join(path.dirname(file), "train_hist.json");
fs.writeFileSync(newFile, JSON.stringify(out, null, 2));
const newCsvFile = path.join(path.dirname(file), "train_hist.csv");
fs.writeFileSync(newCsvFile, csvOut);
console.log(`cleaned json written to ${newFile}`);
