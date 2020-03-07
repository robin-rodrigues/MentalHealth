/*var mongoose = require('mongoose');
var Campground = require('./models/campground');
var Comment = require('./models/comment');

var data = [
    {
        name: "Clouds Rest",
        image: "https://farm8.staticflickr.com/7381/9705573948_3f342901d1.jpg",
        description: "blahblahblah"
    },
    {
        name: "Canyons Floor",
        image: "https://farm4.staticflickr.com/3052/3484099068_3b4b46c0da.jpg",
        description: "blahblahblah"
    },
    {
        name: "Desert Mesa",
        image: "https://farm6.staticflickr.com/5108/5789045796_27c9217bf2.jpg",
        description: "blahblahblah"
    }
]

function seedDB(){
    //Remove all campgrounds
    Campground.remove({},function(err){
     /* if(err){
            console.log(err);
        }
        console.log("All campgrounds removed!")
        //add a few campgrounds
        data.forEach(function(seed){
            Campground.create(seed,function(err,campground){
                if(err){
                    console.log(err)
                }else{
                    console.log("Added a campground!");
                    Comment.create(
                        {
                            text:"This place is great ,but i wish there was internet!",
                            author: "Robin Rodrigues"
                        },function(err,comment){
                            if(err){
                                console.log(err)
                            }else{
                                campground.comments.push(comment);
                                campground.save();
                                console.log("Created a new comment!");
                            }
                        }
                    )
                }
            })
        });
    });
   
}

module.exports = seedDB;*/