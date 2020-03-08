var mongoose = require("mongoose");

var DiagnosticSchema = new mongoose.Schema({
    age: Number,
    gender: Number,
    family_history: Number,
    benefits: Number,
    care_options: Number,
    anonymity: Number,
    leave: Number,
    work_interfere: Number
});

module.exports = mongoose.model("Diagnostic",DiagnosticSchema);