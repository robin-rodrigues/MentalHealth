var mongoose = require("mongoose");
var passportLocalMongoose = require("passport-local-mongoose");
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

DiagnosticSchema.plugin(passportLocalMongoose);

module.exports = mongoose.model("Diagnostic",DiagnosticSchema);