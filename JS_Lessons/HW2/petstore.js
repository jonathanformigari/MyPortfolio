/**
 * This function should calculate the total amount of pet food that should be
 * ordered for the upcoming week.
 * @param numAnimals the number of animals in the store
 * @param avgFood the average amount of food (in kilograms) eaten by the animals
 * 				each week
 * @return the total amount of pet food that should be ordered for the upcoming
 * 				 week, or -1 if the numAnimals or avgFood are less than 0 or non-numeric
 */
function calculateFoodOrder(numAnimals, avgFood) {
    var transNumAnimals = Number(numAnimals);
    var transAvgFood = Number(avgFood);

    if (isNaN(numAnimals) || isNaN(avgFood) || numAnimals < 0 || avgFood < 0 ) {
        return -1;
    }
    else {
        return transAvgFood*transNumAnimals;
    }
}

/**
 * Determines which day of the week had the most nnumber of people visiting the
 * pet store. If more than one day of the week has the same, highest amount of
 * traffic, an array containing the days (in any order) should be returned.
 * (ex. ["Wednesday", "Thursday"]). If the input is null or an empty array, the function
 * should return null.
 * @param week an array of Weekday objects
 * @return a string containing the name of the most popular day of the week if there is only one most popular day, and an array of the strings containing the names of the most popular days if there are more than one that are most popular
 */
function mostPopularDays(week) {
    // IMPLEMENT THIS FUNCTION!

    var flag = 0;
    var mostPopularDay;
    var listOfMostPopularDays = [];

    if (week == null) {
        return null
    }

    if (week.length == 0 || week[0] == null) {
        return null;
    }

    week.sort(function(a, b){return b.traffic-a.traffic});

    listOfMostPopularDays.push(week[0].name);

    for (var day = 1; day < week.length; day++){
        if (week[day].traffic >= week[0].traffic){
            flag = 1;
            listOfMostPopularDays.push(week[day].name);            
        } 
    }

    if (flag == 0) {
        return listOfMostPopularDays[0];
    }
    else {
        return listOfMostPopularDays;
    }
}


/**
 * Given three arrays of equal length containing information about a list of
 * animals - where names[i], types[i], and breeds[i] all relate to a single
 * animal - return an array of Animal objects constructed from the provided
 * info.
 * @param names the array of animal names
 * @param types the array of animal types (ex. "Dog", "Cat", "Bird")
 * @param breeds the array of animal breeds
 * @return an array of Animal objects containing the animals' information, or an
 *         empty array if the array's lengths are unequal or zero, or if any array is null.
 */
function createAnimalObjects(names, types, breeds) {
    var listOfAnimalObjects = [];

    if (names == null || breeds == null || types == null) {
        return listOfAnimalObjects
    }

    if (names.length == 0 || breeds.length == 0 || types.length == 0 || names.length != breeds.length 
        || types.length != breeds.length || names.length != types.length ) {
             return listOfAnimalObjects;
         }

    for (animal = 0; animal < names.length; animal++) {
        listOfAnimalObjects[animal] = new Animal(names[animal], types[animal], breeds[animal]);
    }

    return listOfAnimalObjects;
}

/////////////////////////////////////////////////////////////////
//
//  Do not change any code below here!
//
/////////////////////////////////////////////////////////////////


/**
 * A prototype to create Weekday objects
 */
function Weekday (name, traffic) {
    this.name = name;
    this.traffic = traffic;
}

/**
 * A prototype to create Item objects
 */
function Item (name, barcode, sellingPrice, buyingPrice) {
     this.name = name;
     this.barcode = barcode;
     this.sellingPrice = sellingPrice;
     this.buyingPrice = buyingPrice;
}
 /**
  * A prototype to create Animal objects
  */
function Animal (name, type, breed) {
    this.name = name;
     this.type = type;
     this.breed = breed;
}


/**
 * Use this function to test whether you are able to run JavaScript
 * from your browser's console.
 */
function helloworld() {
    return 'hello world!';
}
