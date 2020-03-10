require('dotenv').config();

var express         = require('express'),
    app             = express(),
    bodyParser      = require('body-parser'),
    mongoose        = require('mongoose'),
    flash           = require('connect-flash'),
    passport        = require('passport'),
    LocalStrategy   = require('passport-local'),
    methodOverrride = require('method-override'),
// Campgrounds     = require('./model/campground'),
 // Comment         = require('./models/comment'),
    User            = require('./model/user');
// Diagnostic     = require('./models/diagnostic')
 //   seedDB          = require('./seeds');

// var CampgroundsRoutes = require("./routes/campgrounds"),
    // CommentsRoutes    = require("./routes/comments"), 
var  IndexRoutes      = require("./routes/index");  

mongoose.connect("mongodb://localhost:27017/yelpcamp",{useNewUrlParser: true});
app.use(bodyParser.urlencoded({extended: true}));
app.set("view engine","ejs");
app.use(express.static(__dirname + "/public"));
app.use(methodOverrride("_method"));
app.use(flash());

//seedDB(); //seed the database

// PASSPORT CONFIG
app.use(require("express-session")({
   secret: "Once again rusty wins!",
   resave:  false,
   saveUninitialzed: false
}));
app.use(passport.initialize());
app.use(passport.session());
passport.use(new LocalStrategy(User.authenticate()));
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

app.use(function(req,res,next){
    res.locals.currentUser = req.user;
    res.locals.error = req.flash("error"); 
    res.locals.success = req.flash("success");
    next();
});

app.use("/", IndexRoutes);
// app.use("/campgrounds", CampgroundsRoutes);
// app.use("/campgrounds/:id/comments/", CommentsRoutes);
app.get('/name', callName); 
  
function callName(req, res) { 
      
    // Use child_process.spawn method from  
    // child_process module and assign it 
    // to variable spawn 
    var spawn = require("child_process").spawn; 
      
    // Parameters passed in spawn - 
    // 1. type_of_script 
    // 2. list containing Path of the script 
    //    and arguments for the script  
      
    // E.g : http://localhost:3000/name?firstname=Mike&lastname=Will 
    // so, first name = Mike and last name = Will 
    var process = spawn('python',["./python.py", 
    req.query.age, 
    req.query.gender,
    req.query.family_history,
    req.query.benefits,
    req.query.care_options,
    req.query.anonymity,
    req.query.leave,
    req.query.work_interfere] );
    // Takes stdout data from script which executed 
    // with arguments and send this data to res object 
    process.stdout.on('data', function(data) { 
        console.log(data.toString());
        console.log(data)
        if(data.toString()==1)
        {
            res.render("dashboard")
        }
        if(data.toString()==0)
        {
            res.render('index')
        }
    } ) 
} 

app.listen(3000,function(){
    console.log(" Server has started");
});
